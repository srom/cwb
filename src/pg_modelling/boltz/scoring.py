import argparse
import json
import logging
from pathlib import Path
import re
import sys

from Bio import SeqIO
import numpy as np
import pandas as pd

from src.pg_modelling.ligand_utils import run_pose_busters, POSEBUSTERS_CHECKS


logger = logging.getLogger(__name__)


def process_boltz_ligand_pulldown_results(
    protein_name : str, 
    results_folder : Path,
) -> pd.DataFrame:
    data = {
        'protein_name': [],
        'ligand_name': [],
        'seed': [],
        'sample': [],
        'structure_file': [],
        'ptm': [],
        'iptm': [],
        'confidence': [],
    }
    for seed_folder in results_folder.iterdir():
        if not seed_folder.is_dir() or not seed_folder.name.startswith('seed_'):
            continue

        seed = int(re.match(r'^seed_([0-9]+)$', seed_folder.name)[1])

        inner_folder = seed_folder / 'boltz_results_inputs' / 'predictions'

        for result_folder in inner_folder.iterdir():
            if not result_folder.is_dir() or not result_folder.name.startswith(protein_name):
                continue

            ligand_name = result_folder.name.split('__')[1]

            models = []
            for f in result_folder.glob('*_model_*.cif'):
                structure_file = f
                score_file = (
                    f.parent / 
                    f"confidence_{f.name.replace('.cif', '.json')}"
                )
                sample = int(re.match(r'^.+_model_([0-9]+)\.cif$', structure_file.name)[1])
                models.append((structure_file, score_file, seed, sample))

            for structure_file, score_file, seed, sample in models:
                with score_file.open() as f:
                    scores = json.load(f)

                ptm = scores['ptm']
                iptm = scores['iptm']
                confidence = np.round(0.8 * iptm + 0.2 * ptm, 3)

                data['protein_name'].append(protein_name)
                data['ligand_name'].append(ligand_name)
                data['seed'].append(seed)
                data['sample'].append(sample)
                data['structure_file'].append(structure_file.as_posix())
                data['ptm'].append(ptm)
                data['iptm'].append(iptm)
                data['confidence'].append(confidence)

    results_df = pd.DataFrame.from_dict(data)

    scores, errors, energy_ratios = [], [], []
    for structure_file_path in results_df['structure_file'].values:
        score, errs, energy_ratio = run_boltz_posebusters(structure_file_path)
        scores.append(score)
        errors.append(errs)
        energy_ratios.append(
            np.round(energy_ratio, 1) if energy_ratio is not None else energy_ratio
        )

    results_df['posebusters_score'] = scores
    results_df['energy_ratio'] = energy_ratios
    results_df['posebusters_errors'] = errors

    results_df = results_df.sort_values(
        ['posebusters_score', 'confidence', 'energy_ratio'], 
        ascending=[False, False, True],
    )

    return results_df.set_index([
        'protein_name',
        'ligand_name',
    ])


def run_boltz_posebusters(structure_path_str : str, ligand_id='LIG'):
    try:
        pose_busters_res = run_pose_busters(
            Path(structure_path_str), 
            ligand_id=ligand_id,
            full_report=True,
        ).iloc[0]
    except ValueError as e:
        logger.error(e)
        return None, None, None

    pose_busters_score = pose_busters_res['score']
    energy_ratio = pose_busters_res['energy_ratio']

    error_str = None
    if not pose_busters_res['perfect_score']:
        errs = []
        for col, val in pose_busters_res.items():
            if col in POSEBUSTERS_CHECKS and not val:
                errs.append(col)
        
        error_str = ','.join(errs)
    
    return pose_busters_score, error_str, energy_ratio


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser(description=f'Scoring script for Boltz')
    parser.add_argument(
        '-i', '--modelling_dir', 
        type=Path,
        required=True,
        help='Path to directory containing the modelling results',
    )
    parser.add_argument(
        '-o', '--output_path', 
        type=Path,
        required=True,
        help='Path to output CSV file',
    )
    args = parser.parse_args()

    modelling_dir = args.modelling_dir
    output_path = args.output_path

    if not modelling_dir.is_dir():
        logger.error(f'Directory does not exist: {modelling_dir}')
        sys.exit(1)

    proteins_fasta_path = modelling_dir.parent.parent / 'proteins.fasta'
    protein_names = [
        protein_record.id 
        for protein_record in SeqIO.parse(proteins_fasta_path, 'fasta')
    ]

    scores = []
    for i, protein_name in enumerate(sorted(protein_names)):
        logger.info(f'Processing protein {protein_name} ({i+1:,} of {len(protein_names):,})')
        score_df = process_boltz_ligand_pulldown_results(protein_name, modelling_dir)
        scores.append(score_df)

    logger.info(f'Exporting to {output_path}')
    scores_df = pd.concat(scores)
    scores_df.to_csv(output_path)
    
    logger.info('DONE')
    sys.exit(0)


if __name__ == '__main__':
    main()
