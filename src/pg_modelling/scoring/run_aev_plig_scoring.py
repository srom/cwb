import argparse
import logging
from pathlib import Path
import random
import tempfile
import subprocess
import sys

import pandas as pd


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser(description=f'Scoring script for AF3')
    parser.add_argument(
        '-i', '--scores_csv_path', 
        type=Path,
        required=True,
        help='Path to CSV file containing the current scores',
    )
    args = parser.parse_args()
    scores_csv_path = args.scores_csv_path

    if not scores_csv_path.is_file():
        logger.error(f'Scores CSV file does not exist: {scores_csv_path}')
        sys.exit(1)

    scores_df = pd.read_csv(scores_csv_path)

    scores_df['unique_id'] = scores_df.apply(
        lambda row: '__'.join([
            row['protein_name'],
            row['ligand_name'],
            row['seed'],
            row['sample'],
        ]),
        axis=1,
    )

    sdf_files, pdb_files = []
    for _, row in scores_df.iterrows():
        cif_path = Path(row['structure_file'])
        ligand_sdf_path = cif_path.parent / cif_path.name.replace('.cif', '_ligand.sdf')
        protein_pdb_path = cif_path.parent / cif_path.name.replace('.cif', '_protein.pdb')

        sdf_files.append(ligand_sdf_path if ligand_sdf_path.is_file() else None)
        pdb_files.append(protein_pdb_path if protein_pdb_path.is_file() else None)

    scores_df['sdf_file'] = sdf_files
    scores_df['pdb_file'] = pdb_files

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_csv_file = tmpdir_path / 'inputs.csv'

        scores_df[
            scores_df['sdf_file'].notnull() &
            scores_df['pdb_file'].notnull()
        ][
            ['unique_id', 'sdf_file', 'pdb_file']
        ].to_csv(
            input_csv_file,
            index=False,
        )

        random_number = int(random.uniform(1, 9999))
        run_name = f'scoring_{str(random_number).zfill(4)}'
        script_path = Path(__file__).parent / 'run_aev_plig_scoring.sh'
        result = subprocess.run(
            [
                'bash', script_path.resolve().as_posix(),
                input_csv_file.resolve().as_posix(),
                run_name,
            ],
            stdout=sys.stdout, 
            stderr=sys.stderr,
        )
    
    if result.returncode != 0:
        sys.exit(1)

    output_csv_path = Path(f'../AEV-PLIG-main/output/predictions/{run_name}_predictions.csv')
    predictions_df = pd.read_csv(output_csv_path, index_col='unique_id')
    scores_df = scores_df.set_index('unique_id')

    scores_df['aev_plig_pK'] = None
    for unique_id in scores_df.index:
        if unique_id in predictions_df.index:
            scores_df.loc[unique_id, 'aev_plig_pK'] = predictions_df.loc['unique_id', 'pK']

    output_csv_path.unlink()

    scores_df.reset_index()[
        [
            c for c in scores_df.columns 
            if c not in ('unique_id', 'sdf_file', 'pdb_file')
        ]
    ].sort_values(
        ['posebusters_score', 'aev_plig_pK', 'confidence', 'energy_ratio'], 
        ascending=[False, False, False, True],
    ).to_csv(
        scores_csv_path, 
        index=False,
    )

    logger.info('DONE')
    sys.exit(0)


if __name__ == '__main__':
    main()

