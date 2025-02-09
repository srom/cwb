DOC = """
This script orchestrates the run of protein-ligand interaction models.

The following models are supported:
- [AlphaFold 3](https://github.com/google-deepmind/alphafold3)
- [Protenix](https://github.com/bytedance/Protenix)
- [Boltz](https://github.com/jwohlwend/boltz)
- [Chai](https://github.com/chaidiscovery/chai-lab)

Inputs:
- Path to fasta file containing protein sequences to be considered
- Path to ligands definition in CSV format (with header):
    - ligand_id: unique identifier for this ligand
    - smiles: ligand definition in SMILES format
- Output directory
- Flags of supported model:
    - any of --af3, --protenix, --boltz, --chai
- Worfklow orchestrator: one of slurm or pbspro
- Name of conda environments:
    - Model environment (one for each model)
    - Post-Processing environment (for scoring)
- Time budget in hours for each stage:
    - MSA generation
    - Modelling
    - Scoring

Logic: 
- Load protein sequences (Biopython)
- Load ligand from SMILES and generate 3D conformation with ETKDGv3 (RDKit)
- Create specific inputs matching format requirements of each requested model
- Generate worklow jobs for the target orchestrator
    - MSA generation step
    - Modelling step
    - Post-processing step (scoring)
- Run worflow with orchestrator.

TODO: should this be a nextflow pipeline instead?
"""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Iterator, List, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from rdkit import Chem

from src.pg_modelling.ligand_utils import (
    generate_conformation, 
    generate_ccd_from_mol,
    sanitize_protein_id,
    sanitize_ligand_name,
    gen_model_seeds,
)


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    args = parse_args()

    proteins_fasta_path = args.proteins_fasta_path
    ligands_csv_path = args.ligands_csv_path
    output_dir = args.output_dir
    orchestrator = args.orchestrator
    models = [t[0] for t in [
        ('af3', args.af3), 
        ('protenix', args.protenix), 
        ('boltz', args.boltz), 
        ('chai', args.chai),
    ] if t[1]]

    if not proteins_fasta_path.is_file():
        logger.error(f'Input fasta file does not exist: {proteins_fasta_path}')
        sys.exit(1)
    elif not ligands_csv_path.is_file():
        logger.error(f'Input CSV file does not exist: {ligands_csv_path}')
        sys.exit(1)
    elif not output_dir.is_dir():
        logger.error(f'Output directory does not exist: {output_dir}')
        sys.exit(1)
    elif len(models) == 0:
        logger.error('Specify at least one of --af3, --protenix, --boltz or --chai')
        sys.exit(1)
    elif args.af3 and orchestrator != 'slurm':
        logger.error('AlphaFold 3 is only supported on SLURM for now')
        sys.exit(1)
    elif (args.protenis or args.boltz or args.chai) and orchestrator != 'pbspro':
        logger.error('Protenix, Bolz and Chai are only supported on PBSPro for now')
        sys.exit(1)

    conda_env_names = {model: getattr(args, f'env_{model}') for model in models}
    time_budget = {
        'msa': args.hours_msa,
        'modelling': args.hours_modelling,
        'scoring': args.hours_scoring,
    }
    n_cpus = args.n_cpus

    logger.info('Running protein-ligand pulldown orchestrator')
    logger.info((
        'Inputs:\n',
        f'\t- Proteins fasta path = {proteins_fasta_path}\n'
        f'\t- Ligands csv path = {ligands_csv_path}\n'
        f'\t- Output dir = {output_dir}\n'
        f'\t- Orchestrator = {orchestrator}\n'
        f'\t- Models = {models}\n'
        f'\t- Conda env names = {conda_env_names}\n'
        f'\t- Time budget (hours) = {time_budget}\n'
        f'\t- Number of CPUs = {n_cpus}\n'
    ))

    proteins = list(SeqIO.parse(proteins_fasta_path, 'fasta'))
    for protein in proteins:
        protein.id = sanitize_protein_id(protein.id)
    
    ligands = pd.read_csv(ligands_csv_path)
    ligands['ligand_id'] = ligands.apply(lambda ligand_id: sanitize_ligand_name(ligand_id))
    ligands = ligands.set_index('ligand_id')

    logger.info(f'Number of sequences: {len(proteins):,}')
    logger.info(f'Number of ligands: {len(ligands):,}')

    # Save inputs to output dir for reference
    ligands.to_csv(output_dir / 'ligands.csv')
    fasta_path = output_dir / 'proteins.fasta'
    with fasta_path.open('w') as f_out:
        SeqIO.write(proteins.values(), f_out, 'fasta')

    # Create MSA folder
    msa_folder = output_dir / 'msa'
    msa_folder.mkdir()

    # Create log folder
    logs_foder = output_dir / 'logs'
    logs_foder.mkdir()

    # Create model output folders
    model_dirs = {}
    for model in models:
        result_dir = output_dir / f'{model}'
        result_dir.mkdir()
        model_dirs[model] = result_dir

    # Generate MSA script
    msa_script_path = generate_msa_script(
        fasta_path, 
        models, 
        msa_folder, 
        logs_foder, 
        orchestrator, 
        max_runtime_in_hours=time_budget['msa'],
    )

    # Generate inputs
    generate_modelling_inputs(proteins, ligands, models, msa_folder, model_dirs, n_cpus)

    # Generate Modelling scripts

    # Generate Scoring scripts

    # Orchestrate 

    # Run

    logger.info('Pulldown started')
    sys.exit(0)


def generate_msa_script(
    fasta_path : Path, 
    models : List[str], 
    msa_folder : Path,
    logs_foder : Path,
    orchestrator : str,
    max_runtime_in_hours : int,
) -> Path:
    if orchestrator == 'pbspro':
        current_path = Path(os.path.abspath(__file__)).parent
        msa_script_path = current_path / 'pbspro' / 'run_msa_search.sh'
        with msa_script_path.open('r') as f:
            msa_script_raw = f.read()
        
        msa_script = msa_script_raw.format(
            input=fasta_path.resolve().as_posix(),
            output=msa_folder.resolve().as_posix(),
            output_log_qsub=logs_foder / 'output_run_msa_search_qsub.log',
            error_log_qsub=logs_foder / 'error_run_msa_search_qsub.log',
            output_log=logs_foder / 'output_run_msa_search.log',
            error_log=logs_foder / 'error_run_msa_search.log',
            time_budget=encode_pbspro_time_budget(max_runtime_in_hours),
            run_protenix_postprocessing='true' if 'protenix' in models else 'false',
            run_chai_postprocessing='true' if 'chai' in models else 'false',
            protenix_postprocessing_script=current_path / 'protenix' / 'run_colabfold_postprocess.py',
            chai_postprocessing_script=current_path / 'chai' / 'run_colabfold_postprocess.py',
        )
    elif orchestrator == 'slurm':
        msa_script_path = Path(os.path.abspath(__file__)) / 'af3' / 'run_msa_search.sh'
        with msa_script_path.open('r') as f:
            msa_script_raw = f.read()

        for protein_record in SeqIO.parse(fasta_path, 'fasta'):
            name = protein_record.id
            data = {
                'name': name,
                'sequences': [{
                    'protein': {
                        'id': 'A',
                        'sequence': str(protein_record.seq).upper()
                    },
                }],
                'modelSeeds': [1],
                'dialect': 'alphafold3',
                'version': 1,
            }
            with (msa_folder / f'{name}.json').open('w') as f_out:
                json.dump(
                    data, 
                    f_out,
                    indent=True,
                )

        msa_script = msa_script_raw.format(
            input=msa_folder.resolve().as_posix(),
            output=msa_folder.resolve().as_posix(),
            log_path=logs_foder / 'af3_msa_run_%j.log',
            time_budget=encode_slurm_time_budget(max_runtime_in_hours),
        )
    else:
        raise ValueError(f'Unknown orchestrator: {orchestrator}')
    
    msa_script_path = (msa_folder / 'run_msa_search.sh')
    with msa_script_path.open('w') as f_out:
        f_out.write(msa_script)

    return msa_script_path


def generate_modelling_inputs(
    proteins : List[SeqRecord], 
    ligands : pd.DataFrame, 
    models : List[str], 
    msa_folder : Path,
    model_dirs : Dict[str, Path], 
    n_cpus : int,
):
    for protein in proteins:
        for ligand_id, ligand_mol in parse_ligands(ligands, n_cpus):
            for model in models:
                model_dir = model_dirs[model]

                if model == 'af3':
                    generate_af3_input(protein, ligand_id, ligand_mol, model_dir, msa_folder)
                elif model == 'protenix':
                    generate_protenix_input(protein, ligand_id, ligand_mol, model_dir, msa_folder)
                elif model == 'boltz':
                    generate_boltz_input(protein, ligand_id, ligand_mol, model_dir, msa_folder)
                elif model == 'chai':
                    generate_chai_input(protein, ligand_id, ligand_mol, model_dir, msa_folder)
                else:
                    raise ValueError(f'Unknown model: {model}')


def generate_af3_input(
    protein : SeqRecord, 
    ligand_id : str, 
    ligand_mol : Chem.Mol, 
    model_dir : Path, 
    msa_folder : Path,
):
    pname = protein.id.lower()
    protein_spec_path = msa_folder / f'{pname}' / f'{pname}_data.json'
    with protein_spec_path.open() as f:
        protein_spec = json.load(f)

    ligand_seq = {
        'ligand': {
            'id': 'B',
            'ccdCodes': [ligand_id],
        }
    }
    try:
        ccd_data = generate_ccd_from_mol(ligand_mol, ligand_id)
    except ValueError:
        print(f'Error for ligand: {ligand_id}')
        raise

    ligands_spec = {
        'sequences': [ligand_seq],
        'userCCD': ccd_data,
    }

    name = f'{protein.id}__{ligand_id}'
    spec = copy.deepcopy(protein_spec)
    spec['name'] = name
    spec['modelSeeds'] = gen_model_seeds(3)
    spec['sequences'] += ligands_spec['sequences']
    spec['userCCD'] = ligands_spec['userCCD']

    output_dir = model_dir / 'inputs'
    output_dir.mkdir(exist_ok=True)

    with (output_dir / f'{name}.json').open('w') as f_out:
        json.dump(spec, f_out, indent=True)


def generate_protenix_input(
    protein : SeqRecord, 
    ligand_id : str, 
    ligand_mol : Chem.Mol, 
    model_dir : Path, 
    msa_folder : Path,
):
    pdb_folder = model_dir / 'ligands'
    pdb_folder.mkdir(exist_ok=True)

    pdb_file = pdb_folder / f'{ligand_id}.pdb'
    Chem.MolToPDBFile(ligand_mol, pdb_file.as_posix())

    precomputed_msa_dir = msa_folder / 'msa' / f'{protein.id}'

    name = f'{protein.id}__{ligand_id}'
    spec = {
        'name': name,
        'sequences': [
            {
                'proteinChain': {
                    'sequence': str(protein.seq).upper(),
                    'count': 1,
                    'msa': {
                        'precomputed_msa_dir': precomputed_msa_dir.resolve().as_posix(),
                        'pairing_db': 'uniref100'
                    }
                }
            },
            {
                'ligand': {
                    'ligand': f'FILE_{pdb_file.resolve().as_posix()}',
                    'count': 1,
                }
            }
        ],
    }

    output_dir = model_dir / 'inputs'
    output_dir.mkdir(exist_ok=True)

    with (output_dir / f'{name}.json').open('w') as f_out:
        json.dump(
            [spec], 
            f_out,
            indent=True,
        )


def generate_boltz_input(
    protein : SeqRecord, 
    ligand_id : str, 
    ligand_mol : Chem.Mol, 
    model_dir : Path, 
    msa_folder : Path,
):
    pass


def generate_chai_input(
    protein : SeqRecord, 
    ligand_id : str, 
    ligand_mol : Chem.Mol, 
    model_dir : Path, 
    msa_folder : Path,
):
    pass


def parse_ligands(ligands : pd.DataFrame, n_cpus : int) -> Iterator[Tuple[str, Chem.Mol]]:
    for ligand_id in ligands.index:
        smiles = ligands.loc[ligand_id, 'smiles']
        yield (
            ligand_id,
            generate_conformation(Chem.MolFromSmiles(smiles), n_cpus=n_cpus),
        )


def encode_pbspro_time_budget(max_runtime_in_hours : int) -> str:
    return f'{str(max_runtime_in_hours).zfill(2)}:00:00',


def encode_slurm_time_budget(max_runtime_in_hours : int) -> str:
    days = max_runtime_in_hours // 24
    hours = max_runtime_in_hours - 24 * days
    return f'{days}-{str(hours).zfill(2)}:00:00'


def parse_args():
    parser = argparse.ArgumentParser(
        description=f'Protein-Ligand pulldown orchestrator.\n\n{DOC}',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-i', '--proteins_fasta_path', 
        type=Path,
        required=True,
        help='Path to fasta file containing protein sequences',
    )
    parser.add_argument(
        '-l', '--ligands_csv_path', 
        type=Path,
        required=True,
        help='Path to CSV file containing ligand definitions',
    )
    parser.add_argument(
        '-o', '--output_dir', 
        type=Path,
        required=True,
        help='Path to output directory. Must exist.',
    )
    parser.add_argument(
        '-w', '--orchestrator', 
        type=str,
        required=True,
        choices=['slurm', 'pbspro'],
        help='Choice of orchestrator: slurm or pbspro',
    )
    parser.add_argument(
        '--af3', 
        action='store_true',
        help='Run AlphaFold 3 (https://github.com/google-deepmind/alphafold3)',
    )
    parser.add_argument(
        '--protenix', 
        action='store_true',
        help='Run Protenix (https://github.com/bytedance/Protenix)',
    )
    parser.add_argument(
        '--boltz', 
        action='store_true',
        help='Run Boltz (https://github.com/jwohlwend/boltz)',
    )
    parser.add_argument(
        '--chai', 
        action='store_true',
        help='Run Chai (https://github.com/chaidiscovery/chai-lab)',
    )
    parser.add_argument(
        '--env_af3', 
        type=str,
        required=False,
        default='af3',
        help='Name of AF3 conda environment',
    )
    parser.add_argument(
        '--env_protenix', 
        type=str,
        required=False,
        default='protenix',
        help='Name of Protenix conda environment',
    )
    parser.add_argument(
        '--env_boltz', 
        type=str,
        required=False,
        default='boltz',
        help='Name of Boltz conda environment',
    )
    parser.add_argument(
        '--env_chai', 
        type=str,
        required=False,
        default='chai',
        help='Name of Chai conda environment',
    )
    parser.add_argument(
        '--env_scoring', 
        type=str,
        required=False,
        default='cwb',
        help='Name of post-processing (scoring) conda environment',
    )
    parser.add_argument(
        '--hours_msa', 
        type=int,
        required=False,
        default=2,
        help='Time budget in hours for MSA generation step',
    )
    parser.add_argument(
        '--hours_modelling', 
        type=int,
        required=False,
        default=2,
        help='Time budget in hours for modelling step',
    )
    parser.add_argument(
        '--hours_scoring', 
        type=int,
        required=False,
        default=2,
        help='Time budget in hours for scoring step',
    )
    parser.add_argument(
        '--n_cpus', 
        type=int,
        required=False,
        default=1,
        help='Number of CPUs to use for generating conformations with RDkit',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
