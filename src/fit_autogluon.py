import argparse
import logging
import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from autogluon.tabular import TabularDataset, TabularPredictor


logger = logging.getLogger()


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
        '-l', '--time_limit', 
        help='Time limit in seconds', 
        type=int,
        default=3600,
    )
    parser.add_argument(
        '-q', '--model_presets', 
        help='Model quality presets from AutoGluon', 
        choices=[
            'medium_quality', 
            'good_quality', 
            'high_quality', 
            'best_quality', 
            'interpretable', 
            'optimize_for_deployment',
        ],
        type=str,
        default='medium_quality',
    )
    parser.add_argument(
        '--test_size', 
        help='Portion of the data left aside as test set', 
        type=float,
        default=0.1,
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
    time_limit = args.time_limit
    model_presets = args.model_presets
    test_size = args.test_size
    random_state = args.random_state
    use_gpu = args.use_gpu
    n_cpus = args.n_cpus

    logger.info('Predicting domain presence')
    logger.info('Fitting with AutoML framework AutoGluon')
    logger.info((
        'Parameters:\n'
        f'-> data folder: {data_folder}\n'
        f'-> output folder: {output_folder}\n'
        f'-> target: {target}\n'
        f'-> time limit (s): {time_limit:,}\n'
        f'-> model presets: {model_presets}\n'
        f'-> test size: {test_size}\n'
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

    if use_gpu and not torch.cuda.is_available():
        logger.error('Option --use_gpu was used but no GPU available - aborting')
        sys.exit(1)
    elif use_gpu:
        logger.info(f'GPU available: {torch.cuda.is_available()}')
        logger.info(f'torch.cuda.device_count() = {torch.cuda.device_count()}')
        n_gpus = 1
    else:
        n_gpus = 0

    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    logger.info('Loading dataset')

    data_df = load_dataset(data_folder)

    logger.info(f'Number of genomes: {len(data_df):,}')
    logger.info(f'Number of variables: {len(data_df.columns):,}')

    if target not in data_df.columns:
        logger.error(f'Target domain not found in dataset: {target}')
        sys.exit(1)
    else:
        n_genomes_with_domain = data_df[target].sum()
        p = 100 * n_genomes_with_domain / len(data_df)
        logger.info(f'Number of genomes with {target}: {n_genomes_with_domain:,} ({p:.0f} %)')

    train_df, test_df = train_test_split(data_df, test_size=test_size)

    train_df.to_csv(output_folder / 'train_set.csv')
    test_df.to_csv(output_folder / 'test_set.csv')

    logger.info('Fitting model')

    train_dataset = TabularDataset(train_df.reset_index(drop=True))
    test_dataset = TabularDataset(test_df.reset_index(drop=True))

    predictor = TabularPredictor(
        label=target,
        problem_type='binary',
        eval_metric='balanced_accuracy',
        path=output_folder,
        log_to_file=True,
        sample_weight='balance_weight',
    )

    predictor.fit(
        train_data=train_dataset,
        time_limit=time_limit,
        presets=model_presets,
        num_cpus=n_cpus,
        num_gpus=n_gpus,
    )

    logger.info('Evaluate on test set')

    res_dict = predictor.evaluate(test_dataset)

    logger.info(f'Evaluation results =\n{res_dict}')

    logger.info('Computing feature importance')

    feature_importance = predictor.feature_importance(test_dataset)

    feature_importance.to_csv(output_folder / 'feature_importance.csv')

    logger.info('DONE')
    sys.exit(0)


def load_dataset(data_folder):
    pfam_df = pd.read_csv(data_folder / 'pfam_bacteria.csv', index_col='assembly_accession')
    TIGR_df = pd.read_csv(data_folder / 'TIGR_bacteria.csv', index_col='assembly_accession')
    phylogeny_df = pd.read_csv(data_folder / 'taxonomy_bacteria.csv', index_col='assembly_accession')

    taxonomic_levels = [
        'gtdb_phylum',
        'gtdb_class',
        'gtdb_order',
        'gtdb_family',
    ]

    data_df = pd.merge(
        pfam_df,
        TIGR_df,
        left_index=True,
        right_index=True,
        how='inner',
    )
    data_df = pd.merge(
        data_df,
        phylogeny_df[taxonomic_levels],
        left_index=True,
        right_index=True,
        how='inner',
    )
    return data_df


if __name__ == '__main__':
    main()
