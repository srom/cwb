import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pg_modelling.ligand_utils import run_pose_busters


def process_protenix_ligand_pulldown_results(
    protein_name : str, 
    results_folder : Path,
    run_posebusters : bool = False,
):
    data = {
        'protein_name': [],
        'ligand_name': [],
        'structure_file': [],
        'seed': [],
        'ptm': [],
        'iptm': [],
        'confidence': [],
    }
    for ligand_folder in results_folder.iterdir():
        if not ligand_folder.name.startswith(protein_name):
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

            seed = seed_folder.name

            score_file = None
            for f in (seed_folder / 'predictions').glob('*_sample_0.json'):
                score_file = f
                break

            structure_file = None
            for f in (seed_folder / 'predictions').glob('*_sample_0.cif'):
                structure_file = f
                break

            if score_file is None:
                raise ValueError(f'No score file for ligand {ligand_name} ({seed_folder})')
            if structure_file is None:
                raise ValueError(f'No structure file for ligand {ligand_name} ({seed_folder})')

            with score_file.open() as f:
                scores = json.load(f)
            
            ptm = scores['ptm']
            iptm = scores['iptm']
            confidence = 0.8 * iptm + 0.2 * ptm

            data['protein_name'].append(protein_name)
            data['ligand_name'].append(ligand_name)
            data['structure_file'].append(structure_file.as_posix())
            data['seed'].append(seed)
            data['ptm'].append(ptm)
            data['iptm'].append(iptm)
            data['confidence'].append(confidence)

    protenix_results_df = pd.DataFrame.from_dict(data).sort_values(
        'confidence', 
        ascending=False,
    )

    if run_posebusters:
        scores, errors = [], []
        for structure_file_path in protenix_results_df['structure_file'].values:
            score, errs = run_protenix_posebusters(structure_file_path)
            scores.append(score)
            errors.append(errs)

        protenix_results_df['posebusters_score'] = scores
        protenix_results_df['posebusters_errors'] = errors

        protenix_results_df = protenix_results_df.sort_values(
            ['posebusters_score', 'confidence'], 
            ascending=False,
        )

    return protenix_results_df.drop_duplicates([
        'protein_name', 
        'ligand_name'
    ]).set_index([
        'protein_name',
        'ligand_name',
    ])


def run_protenix_posebusters(structure_path_str : str):
    pose_busters_res = run_pose_busters(Path(structure_path_str), 'l01').iloc[0]

    pose_busters_score = pose_busters_res['score']

    error_str = None
    if not pose_busters_res['perfect_score']:
        errs = []
        for col, val in pose_busters_res.items():
            if col == 'perfect_score':
                continue
            if isinstance(val, np.bool_) and not val:
                errs.append(col)
        
        error_str = ','.join(errs)
    
    return pose_busters_score, error_str
