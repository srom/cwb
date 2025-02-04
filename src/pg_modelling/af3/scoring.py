import json
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi

from src.pg_modelling.ligand_utils import run_pose_busters


def process_af3_ligand_pulldown_results(
    protein_name : str, 
    results_folder : Path,
    run_posebusters : bool = False,
):
    data = {
        'protein_name': [],
        'ligand_name': [],
        'folder': [],
        'structure_file': [],
        'ptm': [],
        'iptm': [],
        'confidence': [],
    }
    for ligand_folder in results_folder.iterdir():
        ligand_folder_name = ligand_folder.name
        if ligand_folder_name.startswith(protein_name.lower()):
            structure_file = ligand_folder / f'{ligand_folder_name}_model.cif'

            assert structure_file.is_file()

            ligand_name = read_ligand_name_from_mmcif(structure_file)
            assert ligand_name is not None

            with (ligand_folder / f'{ligand_folder_name}_summary_confidences.json').open() as f:
                scores = json.load(f)
            
            ptm = scores['ptm']
            iptm = scores['iptm']
            confidence = 0.8 * iptm + 0.2 * ptm

            data['protein_name'].append(protein_name)
            data['ligand_name'].append(ligand_name)
            data['folder'].append(ligand_folder_name)
            data['structure_file'].append(structure_file.as_posix())
            data['ptm'].append(ptm)
            data['iptm'].append(iptm)
            data['confidence'].append(confidence)

    results_df = pd.DataFrame.from_dict(data).sort_values(
        'confidence', 
        ascending=False,
    ).set_index([
        'protein_name',
        'ligand_name',
    ])

    if run_posebusters:
        scores, errors = [], []
        for ix in results_df.index:
            ligand_name = ix[1]
            structure_file_path = results_df.loc[ix, 'structure_file']
            score, errs = run_af3_posebusters(ligand_name, structure_file_path)
            scores.append(score)
            errors.append(errs)

        results_df['posebusters_score'] = scores
        results_df['posebusters_errors'] = errors

        return results_df.sort_values(
            ['posebusters_score', 'confidence'], 
            ascending=False,
        )
    else:
        return results_df


def run_af3_posebusters(ligand_name : str, structure_path_str : str):
    p = Path(structure_path_str)
    pose_busters_res = run_pose_busters(p, ligand_name).iloc[0]
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


def read_ligand_name_from_mmcif(mmcif_file : Path):
    doc = gemmi.cif.read_file(mmcif_file.as_posix())
    block = doc.sole_block()
    return block.find_value('_pdbx_nonpoly_scheme.mon_id')
