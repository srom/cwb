import argparse
import logging
import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch


logger = logging.getLogger()


class LogisticRegression(torch.nn.Module):

    def __init__(self, n_inputs):
        super().__init__()

        self.linear = torch.nn.Linear(n_inputs, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.distributions.ContinuousBernoulli(logits=logits)


def compute_loss(model, x, y, class_weights, lambda_l2=0.0):
    """
    Loss: negative log likelihood of the continuous Bernouilli distribution.

    `class_weights` is a tensor containing the class weights, 
    where `class_weights[0]` is the weight for class 0 and `class_weights[1]` is the weight for class 1.
    """
    bernouilli_dist = model(x)
    log_likelihood = bernouilli_dist.log_prob(y)

    weights = y * class_weights[1] + (1 - y) * class_weights[0]
    weighted_log_likelihood = weights * log_likelihood

    l2_reg = sum(torch.norm(param, p=2)**2 for param in model.parameters())

    loss = -weighted_log_likelihood.mean() + lambda_l2 * l2_reg

    return bernouilli_dist, loss


def load_dataset(data_folder):
    pfam_df = pd.read_csv(data_folder / 'pfam_bacteria.csv', index_col='assembly_accession')
    TIGR_df = pd.read_csv(data_folder / 'TIGR_bacteria.csv', index_col='assembly_accession')
    phylogeny_df = pd.read_csv(data_folder / 'taxonomy_order_bacteria.csv', index_col='assembly_accession')

    data_df = pd.merge(
        pfam_df,
        TIGR_df,
        left_index=True,
        right_index=True,
        how='inner',
    )
    data_df = pd.merge(
        data_df,
        phylogeny_df,
        left_index=True,
        right_index=True,
        how='inner',
    )
    return data_df


def compute_metrics(output, actuals, decision_threshold=0.5):
    probs = output.mean.detach().numpy()
    pred_binary = [1 if x >= decision_threshold else 0 for x in probs]
    return accuracy_score(actuals.detach().numpy(), pred_binary)


def save_model(model, output_folder):
    torch.save(model.state_dict(), output_folder / 'model_weights.pth')


def save_coefficients(parameters, coefficients, output_folder):
    pd.DataFrame.from_dict({
        'parameters': parameters,
        'coefficient': coefficients,
    }).sort_values(
        'coefficient',
        ascending=False,
    ).to_csv(
        output_folder / 'coefficients.csv',
        index=False,
    )


def parameters_grid_search(model, X_train, y_train, class_weights, test_size=0.1):
    learning_rates = [2, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    l2_lambdas = [1e-7]

    n_cells = len(learning_rates) * len(l2_lambdas)
    temp_dir = Path(tempfile.mkdtemp())

    n = 0
    scored_params = []
    for learning_rate in learning_rates:
        for lambda_l2 in l2_lambdas:
            n += 1
            logger.info(f'Grid cell {n:,} / {n_cells:,}: lr = {learning_rate} | lamda L2 = {lambda_l2}')

            output_folder = temp_dir / f'grid_cell_{str(n).zfill(4)}'
            output_folder.mkdir()

            loss_fn = lambda model, x, y: compute_loss(model, x, y, class_weights, lambda_l2)

            X_grid_train, X_grid_val, y_grid_train, y_grid_val = train_test_split(
                X_train, 
                y_train, 
                test_size=test_size,
            )

            optimizer = torch.optim.LBFGS(
                model.parameters(), 
                lr=learning_rate,
                max_iter=10,
            )

            def training_loop():
                model.train()
                optimizer.zero_grad()
                _, loss = loss_fn(model, X_grid_train, y_grid_train)
                loss.backward()
                return loss
            
            optimizer.step(training_loop)

            with torch.no_grad():
                model.eval()

                val_dist, _ = loss_fn(model, X_grid_val, y_grid_val)
                val_accuracy = np.round(100 * compute_metrics(val_dist, y_grid_val), 2)

            logger.info(f'Grid Cell {n:,} / {n_cells:,}: lr = {learning_rate} | lambda L2 = {lambda_l2} | accuracy = {val_accuracy:.2f} %')

            scored_params.append((
                learning_rate,
                lambda_l2,
                val_accuracy,
            ))
            
    best_params = sorted(scored_params, key=lambda t: t[-1], reverse=True)[0]
    learning_rate, lambda_l2, best_accuracy = best_params

    logger.info(f'Best parameters: lr = {learning_rate} | lambda L2 = {lambda_l2} | accuracy = {best_accuracy:.2f} %')

    return learning_rate, lambda_l2


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data_folder', 
        help='Path to data folder', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o', '--output_folder', 
        help='Path to output folder', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-t', '--target', 
        help='Target domain (Pfam short name or TIGR id)', 
        type=str,
        required=True,
    )
    parser.add_argument(
        '-n', '--n_epochs', 
        help='Number of epochs', 
        type=int,
        required=True,
    )
    parser.add_argument(
        '-m', '--max_iter_per_epoch', 
        help='Max number of iterations per epoch', 
        type=int,
        default=10,
    )
    parser.add_argument(
        '-l', '--learning_rate', 
        help='Learning rate', 
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '-c', '--use_class_weights', 
        help='Deal with class imbalance with class weights', 
        action='store_true',
    )
    parser.add_argument(
        '--l2', 
        help='L2 regularization', 
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '--grid_search', 
        help='Perform hyperparameter grid search', 
        action='store_true',
    )
    parser.add_argument(
        '--random_state', 
        help='Random state for reproducibility', 
        type=int,
        default=None,
    )
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--n_cpus', type=int, default=4)
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder
    target = args.target
    n_epochs = args.n_epochs
    max_iter_per_epoch = args.max_iter_per_epoch
    learning_rate = args.learning_rate
    use_class_weights = args.use_class_weights
    lambda_l2 = args.l2
    grid_search = args.grid_search
    random_state = args.random_state
    use_gpu = args.use_gpu
    n_cpus = args.n_cpus

    if not data_folder.is_dir():
        logger.error(f'Data folder does not exist: {data_folder}')
        sys.exit(1)
    if not output_folder.is_dir():
        logger.error(f'Output folder does not exist: {output_folder}')
        sys.exit(1)

    if use_gpu and not torch.cuda.is_available():
        logger.error('Option --use_gpu was used but no GPU available - aborting')
        sys.exit(1)
    elif use_gpu:
        logger.info(f'GPU available: {torch.cuda.is_available()}')
        logger.info(f'torch.cuda.device_count() = {torch.cuda.device_count()}')

    torch.set_num_threads(n_cpus)

    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    logger.info('Loading dataset')
    
    data_df = load_dataset(data_folder)

    logger.info(f'Number of genomes: {len(data_df):,}')
    logger.info(f'Number of variables: {len(data_df.columns):,}')

    parameters = [c for c in data_df.columns if c != target]

    if target not in data_df.columns:
        logger.error(f'Target domain not found in dataset: {target}')
        sys.exit(1)
    else:
        n_genomes_with_domain = data_df[target].sum()
        p = 100 * n_genomes_with_domain / len(data_df)
        logger.info(f'Number of genomes with {target}: {n_genomes_with_domain:,} ({p:.0f} %)')

    response = torch.tensor(data_df[target].astype(np.float32).values)
    predictors = torch.tensor(data_df.loc[:, data_df.columns != target].astype(np.float32).values)

    X_train, X_val, y_train, y_val = train_test_split(
        predictors, 
        response, 
        test_size=0.1,
    )

    if use_class_weights:
        class_weights = torch.tensor(
            compute_class_weight(
                class_weight='balanced', 
                classes=np.array([0.0, 1.0]), 
                y=y_train.detach().numpy()
            ),
            dtype=torch.float,
        )
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)

    logger.info('Fitting logistic regression with L-BFGS')

    n_inputs = predictors.shape[1]
    model = LogisticRegression(n_inputs)

    if grid_search:
        logger.info('Grid searching for best hyperparameters')
        learning_rate, lambda_l2 = parameters_grid_search(model, X_train, y_train, class_weights)

    loss_fn = lambda model, x, y: compute_loss(model, x, y, class_weights, lambda_l2)

    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=learning_rate,
        max_iter=max_iter_per_epoch,
        history_size=max_iter_per_epoch,
    )

    def training_loop():
        model.train()
        optimizer.zero_grad()
        _, loss = loss_fn(model, X_train, y_train)
        loss.backward()
        return loss
    
    for epoch in range(n_epochs):
        optimizer.step(training_loop)

        with torch.no_grad():
            model.eval()

            train_dist, train_loss = loss_fn(model, X_train, y_train)
            train_accuracy = np.round(100 * compute_metrics(train_dist, y_train), 2)

            val_dist, val_loss = loss_fn(model, X_val, y_val)
            val_accuracy = np.round(100 * compute_metrics(val_dist, y_val), 2)

            logger.info((
                f'Epoch {epoch+1:,}: '
                f'train loss = {train_loss.item()} | '
                f'train accuracy = {train_accuracy.item():.2f} | '
                f'val loss = {val_loss.item()} | '
                f'val accuracy = {val_accuracy:.2f}'
            ))

    logger.info('Saving model')

    coefficients = np.round(model.linear.weight.view(-1).cpu().detach().numpy(), 8)

    save_model(model, output_folder)
    save_coefficients(parameters, coefficients, output_folder)

    logger.info('DONE')
    sys.exit(0)


if __name__ == '__main__':
    main()
