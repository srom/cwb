import json
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi

from src.pg_modelling.ligand_utils import run_pose_busters, POSEBUSTERS_CHECKS


def process_af3_ligand_pulldown_results(
    protein_name : str, 
    results_folder : Path,
    run_posebusters : bool = False,
    score_all_sample : bool = False,
):
    data = {
        'protein_name': [],
        'ligand_name': [],
        'structure_file': [],
        'ptm': [],
        'iptm': [],
        'confidence': [],
    }
    for ligand_folder in results_folder.iterdir():
        ligand_folder_name = ligand_folder.name
        if ligand_folder_name.startswith(protein_name.lower()):
            main_structure_file = ligand_folder / f'{ligand_folder_name}_model.cif'
            assert main_structure_file.is_file()

            ligand_name = read_ligand_name_from_mmcif(main_structure_file)
            assert ligand_name is not None

            main_score_file = ligand_folder / f'{ligand_folder_name}_summary_confidences.json'
            assert main_score_file.is_file()

            models = []
            if not score_all_sample:
                models.append((main_structure_file, main_score_file))
            else:
                for directory in ligand_folder.glob('seed-*_sample-*'):
                    if directory.is_dir():
                        models.append((
                            directory / 'model.cif',
                            directory / 'summary_confidences.json',
                        ))

            for structure_file, score_file in models:
                with score_file.open() as f:
                    scores = json.load(f)
                
                ptm = scores['ptm']
                iptm = scores['iptm']
                confidence = np.round(0.8 * iptm + 0.2 * ptm, 3)

                data['protein_name'].append(protein_name)
                data['ligand_name'].append(ligand_name)
                data['structure_file'].append(structure_file.as_posix())
                data['ptm'].append(ptm)
                data['iptm'].append(iptm)
                data['confidence'].append(confidence)

    results_df = pd.DataFrame.from_dict(data).sort_values(
        'confidence', 
        ascending=False,
    )

    if run_posebusters:
        scores, errors, energy_ratios = [], [], []
        for ligand_name, structure_file_path in results_df[['ligand_name', 'structure_file']].values:
            score, errs, energy_ratio = run_af3_posebusters(ligand_name, structure_file_path)
            scores.append(score)
            errors.append(errs)
            energy_ratios.append(np.round(energy_ratio, 1))

        results_df['posebusters_score'] = scores
        results_df['posebusters_errors'] = errors
        results_df['energy_ratio'] = energy_ratios

        results_df = results_df.sort_values(
            ['posebusters_score', 'energy_ratio', 'confidence'], 
            ascending=[False, True, False],
        )
    
    return results_df.drop_duplicates([
        'protein_name', 
        'ligand_name'
    ]).set_index([
        'protein_name',
        'ligand_name',
    ])


def run_af3_posebusters(ligand_name : str, structure_path_str : str):
    pose_busters_res = run_pose_busters(
        Path(structure_path_str), 
        ligand_id=ligand_name,
        full_report=True,
    ).iloc[0]
    
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


def read_ligand_name_from_mmcif(mmcif_file : Path):
    doc = gemmi.cif.read_file(mmcif_file.as_posix())
    block = doc.sole_block()
    return block.find_value('_pdbx_nonpoly_scheme.mon_id')
