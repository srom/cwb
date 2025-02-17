import argparse
import json
import logging
from pathlib import Path
import re
import sys

from Bio import SeqIO
import numpy as np
import pandas as pd

from src.pg_modelling.ligand_utils import (
    extract_protein_and_ligand_from_mmcif,
    run_pose_busters_from_ligand_and_protein, 
    POSEBUSTERS_CHECKS,
)


logger = logging.getLogger(__name__)


def process_chai_ligand_pulldown_results(
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
    for result_folder in results_folder.iterdir():
        if not result_folder.is_dir() or not result_folder.name.startswith(protein_name):
            continue

        ligand_name = result_folder.name.split('__')[1]

        for seed_folder in result_folder.iterdir():
            if not seed_folder.is_dir() or not seed_folder.name.startswith('seed_'):
                continue

            seed = int(re.match(r'^seed_([0-9]+)$', seed_folder.name)[1])

            models = []
            for f in seed_folder.glob('pred.model_idx_*.cif'):
                structure_file = f
                score_file = (
                    f.parent / 
                    f.name.replace('pred', 'scores').replace('.cif', '.npz')
                )
                sample = int(re.match(r'^pred\.model_idx_([0-9]+)\.cif$', structure_file.name)[1])
                models.append((structure_file, score_file, seed, sample))

            for structure_file, score_file, seed, sample in models:
                with np.load(score_file) as score_data:
                    scores = dict(score_data)

                ptm = scores['ptm'][0]
                iptm = scores['iptm'][0]
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
        score, errs, energy_ratio = run_chai_posebusters(structure_file_path)
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


def run_chai_posebusters(structure_path : Path, ligand_id='LIG2'):
    try:
        ligand_sdf_path = structure_path.parent / structure_path.name.replace('.cif', '_ligand.sdf')
        protein_pdb_path = structure_path.parent / structure_path.name.replace('.cif', '_protein.pdb')
        extract_protein_and_ligand_from_mmcif(
            structure_path,
            protein_pdb_path,
            ligand_sdf_path,
        )
        pose_busters_res = run_pose_busters_from_ligand_and_protein(
            ligand_sdf_path,
            protein_pdb_path,
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

    parser = argparse.ArgumentParser(description=f'Scoring script for Chai')
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
        score_df = process_chai_ligand_pulldown_results(protein_name, modelling_dir)
        scores.append(score_df)

    logger.info(f'Exporting to {output_path}')
    scores_df = pd.concat(scores)
    scores_df.to_csv(output_path)
    
    logger.info('DONE')
    sys.exit(0)


if __name__ == '__main__':
    main()
