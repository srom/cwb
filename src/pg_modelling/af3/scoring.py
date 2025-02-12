import argparse
import json
import logging
from pathlib import Path
import re
import sys

from Bio import SeqIO
import numpy as np
import pandas as pd
import gemmi

from src.pg_modelling.ligand_utils import run_pose_busters, POSEBUSTERS_CHECKS


logger = logging.getLogger(__name__)


def process_af3_ligand_pulldown_results(
    protein_name : str, 
    results_folder : Path,
    run_posebusters : bool = False,
    score_all_sample : bool = False,
    keep_all : bool = False,
):
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
                models.append((main_structure_file, main_score_file, None, None))
            else:
                for directory in ligand_folder.glob('seed-*_sample-*'):
                    if directory.is_dir():
                        structure_file = directory / 'model.cif'

                        seed = int(re.match(r'^seed-([0-9]+)_.+$', directory.name)[1])
                        sample = int(re.match(r'^.+_sample-([0-9]+)$', directory.name)[1])

                        models.append((
                            structure_file,
                            directory / 'summary_confidences.json',
                            seed,
                            sample,
                        ))

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
        results_df['energy_ratio'] = energy_ratios
        results_df['posebusters_errors'] = errors

        results_df = results_df.sort_values(
            ['posebusters_score' 'confidence', 'energy_ratio',], 
            ascending=[False, False, True],
        )

    if not keep_all:
        results_df = results_df.drop_duplicates([
            'protein_name', 
            'ligand_name'
        ])
    
    return results_df.set_index([
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


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser(description=f'Scoring script for AF3')
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
    for i, protein_name in enumerate(protein_names):
        logger.info(f'Processing protein {protein_name} ({i+1:,} of {len(protein_names):,})')
        score_df = process_af3_ligand_pulldown_results(
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
