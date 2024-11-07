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
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


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


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


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


def save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_folder):
    np.save(output_folder / 'train_loss.npy', np.array(train_losses))
    np.save(output_folder / 'val_loss.npy', np.array(val_losses))
    np.save(output_folder / 'train_accuracy.npy', np.array(train_accuracies))
    np.save(output_folder / 'val_accuracy.npy', np.array(val_accuracies))


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_folder):
    sns.set_theme(palette='colorblind', font_scale=1.3)
    palette = sns.color_palette().as_hex()

    plt.ioff()

    f, axes = plt.subplots(2, 1, figsize=(12, 10))

    ax1, ax2 = axes.flatten()

    epochs = list(range(1, len(train_losses) + 1))

    ax1.plot(epochs, train_losses, '-', color=palette[0])
    ax1.plot(epochs, val_losses, '-', color=palette[1])
    ax1.set_ylabel('Loss');

    ax2.plot(epochs, train_accuracies, '-', color=palette[0])
    ax2.plot(epochs, val_accuracies, '-', color=palette[1])
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Accuracy (%)');

    f.savefig(output_folder / 'metrics.png', dpi=300);
    plt.close()


def fit_logistic_regression(
    model, 
    n_epochs, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    learning_rate, 
    batch_size, 
    loss_fn,
    output_folder,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = SimpleDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_train_losses = []
        epoch_train_accuracies = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            train_dist, loss = loss_fn(model, X_batch, y_batch)
            loss.backward()
            optimizer.step()

            train_accuracy = compute_metrics(train_dist, y_batch)

            epoch_train_losses.append(float(loss.detach().numpy()))
            epoch_train_accuracies.append(train_accuracy)

        mean_train_accuracy = np.round(100 * np.mean(epoch_train_accuracies), 2)
        mean_train_loss = np.round(np.mean(epoch_train_losses), 6)

        train_losses.append(mean_train_loss)
        train_accuracies.append(mean_train_accuracy)

        model.eval()
        dist, val_loss = loss_fn(model, X_val, y_val)

        val_accuracy = compute_metrics(dist, y_val)

        val_accuracies.append(np.round(100 * val_accuracy, 2))
        val_losses.append(np.round(val_loss.detach().numpy(), 6))

        logger.info(f'Epoch {str(epoch).zfill(3)} done | validation loss = {val_losses[-1]:.6f} | accuracy = {val_accuracies[-1]} %')

        save_model(model, output_folder)
        save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_folder)

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_folder)

    return model


def compute_coefficient_p_values(model, x, y, loss_fn):
    n_inputs = x.shape[1]

    # Define the negative log likelihood loss
    def negative_log_likelihood(model, x, y):
        return loss_fn(model, x, y)[1]

    # Compute the Hessian matrix of the loss with respect to the model parameters
    def compute_hessian(model, loss_fn, x, y):
        def closure_fn(params):
            model.linear.weight.data = params[:n_inputs].view_as(model.linear.weight)
            model.linear.bias.data = params[n_inputs:]
            return loss_fn(model, x, y)
        
        params = torch.cat([model.linear.weight.view(-1), model.linear.bias])
        hessian_matrix = torch.autograd.functional.hessian(closure_fn, params)
        return hessian_matrix

    # Get the trained model parameters
    params = torch.cat([model.linear.weight.view(-1), model.linear.bias])

    # Compute the Hessian matrix
    logger.info('Compute Hessian matrix')
    hessian_matrix = compute_hessian(model, negative_log_likelihood, x, y)

    # Add a small regularization term to the Hessian matrix diagonal to make it invertible
    lambda_identity = 1e-12 * torch.eye(hessian_matrix.size(0))
    regularized_hessian = hessian_matrix + lambda_identity

    # Compute the standard errors
    # The diagonal elements of the Hessian matrix inverse give the variances
    logger.info('Compute inverse of hessian matrix')
    inv_hessian = torch.inverse(regularized_hessian)
    standard_errors = torch.sqrt(torch.diag(inv_hessian))

    # Compute the z-scores (coefficients / standard errors)
    z_scores = params / standard_errors

    # Compute the p-values from the z-scores
    logger.info('Compute p values')
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores.cpu().detach().numpy())))

    # Remove bias term, round to 8 decimals
    coefficients = np.round(np.array(params.cpu().detach().numpy())[:-1], 8)
    p_values = np.round(np.array(p_values)[:-1], 8)
    
    return coefficients, p_values


def save_coefficients(parameters, coefficients, p_values, output_folder):
    pd.DataFrame.from_dict({
        'parameters': parameters,
        'coefficient': coefficients,
        'p_value': p_values,
    }).sort_values(
        ['coefficient', 'p_value'],
        ascending=[False, True],
    ).to_csv(
        output_folder / 'coefficients.csv',
        index=False,
    )


def parameters_grid_search(model, X_train, y_train, class_weights, n_epochs=20, n_average=5, test_size=0.1):
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    batch_sizes = [8, 16, 32, 64, 128]
    l2_lambdas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    n_cells = len(learning_rates) * len(batch_sizes) * len(l2_lambdas)

    temp_dir = Path(tempfile.mkdtemp())

    n = 0
    scored_params = []
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for lambda_l2 in l2_lambdas:
                n += 1
                logger.info(f'Grid cell {n:,} / {n_cells:,}: lr = {learning_rate} | batch = {batch_size} | lamda L2 = {lambda_l2}')

                output_folder = temp_dir / f'grid_cell_{str(n).zfill(4)}'
                output_folder.mkdir()

                loss_fn = lambda model, x, y: compute_loss(model, x, y, class_weights, lambda_l2)

                X_grid_train, X_grid_val, y_grid_train, y_grid_val = train_test_split(
                    X_train, 
                    y_train, 
                    test_size=test_size,
                )

                fit_logistic_regression(
                    model,
                    n_epochs,
                    X_grid_train,
                    y_grid_train,
                    X_grid_val,
                    y_grid_val,
                    learning_rate,
                    batch_size,
                    loss_fn,
                    output_folder,
                )

                val_accuracies = np.load(output_folder / 'val_accuracy.npy')
                last_accuracies = val_accuracies[-n_average:]
                scoring_accuracy = np.mean(last_accuracies) - np.std(last_accuracies)

                scored_params.append((
                    learning_rate,
                    batch_size,
                    lambda_l2,
                    scoring_accuracy
                ))
    
    best_params = sorted(scored_params, key=lambda t: t[-1], reverse=True)[0]

    learning_rate, batch_size, lambda_l2, best_accuracy = best_params

    logger.info(f'Best parameters: lr = {learning_rate} | batch = {batch_size} | lambda L2 = {lambda_l2} | accuracy = {best_accuracy:.2f} %')

    return learning_rate, batch_size, lambda_l2



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
        '-l', '--learning_rate', 
        help='Learning rate', 
        type=float,
        default=None,
    )
    parser.add_argument(
        '-b', '--batch_size', 
        help='Batch size', 
        type=int,
        default=None,
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
        '--test_size', 
        help='Portion of the data left aside as test set', 
        type=float,
        default=0.1,
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
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    use_class_weights = args.use_class_weights
    lambda_l2 = args.l2
    test_size = args.test_size
    grid_search = args.grid_search
    random_state = args.random_state
    use_gpu = args.use_gpu
    n_cpus = args.n_cpus

    logger.info('Logistic Regression fitting with PyTorch')
    logger.info((
        'Parameters:\n'
        f'-> data folder: {data_folder}\n'
        f'-> output folder: {output_folder}\n'
        f'-> target: {target}\n'
        f'-> n epochs: {n_epochs:,}\n'
        f'-> learning rate: {learning_rate}\n'
        f'-> batch size: {batch_size}\n'
        f'-> use class weights: {use_class_weights}\n'
        f'-> L2 reg: {lambda_l2}\n'
        f'-> test size: {test_size}\n'
        f'-> grid search: {grid_search}\n'
        f'-> random state: {random_state}\n'
        f'-> use gpu: {use_gpu}\n'
        f'-> n cpus: {n_cpus}'
    ))

    if not data_folder.is_dir():
        logger.error(f'Data folder does not exist: {data_folder}')
        sys.exit(1)
    if not output_folder.is_dir():
        logger.error(f'Output folder does not exist: {output_folder}')
        sys.exit(1)

    if not grid_search and (learning_rate is None or batch_size is None):
        logger.error(f'Learning rate and batch size must be set')
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
        test_size=test_size,
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

    n_inputs = predictors.shape[1]
    model = LogisticRegression(n_inputs)

    if grid_search:
        logger.info('Grid searching for best parameters')
        learning_rate, batch_size, lambda_l2 = parameters_grid_search(
            model,
            X_train,
            y_train,
            class_weights,
        )

    logger.info('Fitting logistic regression')

    loss_fn = lambda model, x, y: compute_loss(model, x, y, class_weights, lambda_l2)

    fit_logistic_regression(
        model,
        n_epochs,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate,
        batch_size,
        loss_fn,
        output_folder,
    )

    logger.info('Compute coefficient p_values')
    coefficients, p_values = compute_coefficient_p_values(model, X_train, y_train, loss_fn)
    save_coefficients(parameters, coefficients, p_values, output_folder)

    logger.info('DONE')
    sys.exit(0)


if __name__ == '__main__':
    main()
