import argparse
import json
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from src.pg_modelling.ligand_utils import (
    extract_protein_and_ligand_from_mmcif,
    run_pose_busters_from_ligand_and_protein, 
    POSEBUSTERS_CHECKS,
)


logger = logging.getLogger(__name__)


def process_protenix_ligand_pulldown_results(
    protein_name : str, 
    results_folder : Path,
    run_posebusters : bool = False,
    score_all_sample : bool = False,
    keep_all : bool = False,
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
    for ligand_folder in results_folder.iterdir():
        if not ligand_folder.is_dir() or not ligand_folder.name.startswith(protein_name):
            continue

        ligand_folder_name = ligand_folder.name
        try:
            ligand_name = ligand_folder_name.split('__')[1]
        except IndexError:
            print(ligand_folder_name)
            raise

        for seed_folder in ligand_folder.iterdir():
            if not seed_folder.name.startswith('seed_'):
                continue

            models = []
            pattern = '*_sample_*.cif' if score_all_sample else '*_sample_0.cif'
            for f in (seed_folder / 'predictions').glob(pattern):
                structure_file = f
                score_file = (
                    f.parent / 
                    f.name.replace('.cif', '.json').replace('_sample_', '_summary_confidence_sample_')
                )
                seed = int(re.match(r'^.+_seed_([0-9]+)_.+$', structure_file.name)[1])
                sample = int(re.match(r'^.+_sample_([0-9]+)\.cif$', structure_file.name)[1])
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

    protenix_results_df = pd.DataFrame.from_dict(data).sort_values(
        'confidence', 
        ascending=False,
    )

    if run_posebusters:
        scores, errors, energy_ratios = [], [], []
        for structure_file_path in protenix_results_df['structure_file'].values:
            score, errs, energy_ratio = run_protenix_posebusters(structure_file_path)
            scores.append(score)
            errors.append(errs)
            energy_ratios.append(
                np.round(energy_ratio, 1) if energy_ratio is not None else energy_ratio
            )

        protenix_results_df['posebusters_score'] = scores
        protenix_results_df['energy_ratio'] = energy_ratios
        protenix_results_df['posebusters_errors'] = errors

        protenix_results_df = protenix_results_df.sort_values(
            ['posebusters_score', 'confidence', 'energy_ratio'], 
            ascending=[False, False, True],
        )

    if not keep_all:
        protenix_results_df = protenix_results_df.drop_duplicates([
            'protein_name', 
            'ligand_name'
        ])

    return protenix_results_df.set_index([
        'protein_name',
        'ligand_name',
    ])


def run_protenix_posebusters(structure_path_str : str, ligand_id='l01'):
    try:
        ligand_sdf_path = structure_path_str.parent / structure_path_str.name.replace('.cif', '_ligand.sdf')
        protein_pdb_path = structure_path_str.parent / structure_path_str.name.replace('.cif', '_protein.pdb')
        extract_protein_and_ligand_from_mmcif(
            structure_path_str,
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

    parser = argparse.ArgumentParser(description=f'Scoring script for Protenix')
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
    
    protein_names = set()
    for p in modelling_dir.glob('*__*'):
        if not p.is_dir():
            continue

        protein_name = p.name.split('__')[0]
        protein_names.add(protein_name)

    scores = []
    for i, protein_name in enumerate(sorted(protein_names)):
        logger.info(f'Processing protein {protein_name} ({i+1:,} of {len(protein_names):,})')
        score_df = process_protenix_ligand_pulldown_results(
            protein_name, 
            modelling_dir, 
            run_posebusters=True, 
            score_all_sample=True,
            keep_all=True,
        )
        scores.append(score_df)

    logger.info(f'Exporting to {output_path}')
    scores_df = pd.concat(scores)
    scores_df.to_csv(output_path)

    logger.info('DONE')
    sys.exit(0)


if __name__ == '__main__':
    main()
