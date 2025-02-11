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
- Flags to prevent running certain stages: --skip_msa, --skip_modelling, --skip_scoring
- Number of predictions (~ number of random seeds)
- Worfklow orchestrator: one of slurm or pbspro
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
import json
import logging
import os
from pathlib import Path
import random
import subprocess
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
    skip_msa = args.skip_msa
    skip_modelling = args.skip_modelling
    skip_scoring = args.skip_scoring
    n_predictions = args.n_predictions

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
    elif (args.protenix or args.boltz or args.chai) and orchestrator != 'pbspro':
        logger.error('Protenix, Bolz and Chai are only supported on PBSPro for now')
        sys.exit(1)

    time_budget = {
        'msa': args.hours_msa,
        'modelling': args.hours_modelling,
        'scoring': args.hours_scoring,
    }
    n_cpus = args.n_cpus

    logger.info('Running protein-ligand pulldown orchestrator')
    logger.info((
        'Inputs:\n'
        f'\t- Proteins fasta path = {proteins_fasta_path}\n'
        f'\t- Ligands csv path = {ligands_csv_path}\n'
        f'\t- Output dir = {output_dir}\n'
        f'\t- Orchestrator = {orchestrator}\n'
        f'\t- Models = {models}\n'
        f'\t- Skip MSA step = {skip_msa}\n'
        f'\t- Skip modelling step = {skip_modelling}\n'
        f'\t- Skip scoring step = {skip_scoring}\n'
        f'\t- Number of predictions = {n_predictions}\n'
        f'\t- Time budget (hours) = {time_budget}\n'
        f'\t- Number of CPUs = {n_cpus}\n'
    ))

    proteins = list(SeqIO.parse(proteins_fasta_path, 'fasta'))
    for protein in proteins:
        protein.id = sanitize_protein_id(protein.id)
    
    ligands = pd.read_csv(ligands_csv_path)
    ligands['ligand_id'] = ligands['ligand_id'].apply(lambda ligand_id: sanitize_ligand_name(ligand_id))
    ligands = ligands.set_index('ligand_id')

    logger.info(f'Number of sequences: {len(proteins):,}')
    logger.info(f'Number of ligands: {len(ligands):,}')

    # Save inputs to output dir for reference
    ligands.to_csv(output_dir / 'ligands.csv')
    fasta_path = output_dir / 'proteins.fasta'
    with fasta_path.open('w') as f_out:
        SeqIO.write(proteins, f_out, 'fasta')

    # Create MSA folder
    msa_folder = output_dir / 'msa'
    msa_folder.mkdir(exist_ok=True)

    # Create log folder
    logs_foder = output_dir / 'logs'
    logs_foder.mkdir(exist_ok=True)

    # Create model output folders
    model_dirs = {}
    for model in models:
        result_dir = output_dir / f'{model}'
        result_dir.mkdir(exist_ok=True)
        model_dirs[model] = result_dir

    msa_script_path = None
    if not skip_msa:
        # Generate MSA script
        logger.info('Generating MSA script')
        msa_script_path = generate_msa_script(
            fasta_path, 
            models, 
            msa_folder, 
            logs_foder, 
            orchestrator, 
            max_runtime_in_hours=time_budget['msa'],
        )

    modelling_script_paths = []
    if not skip_modelling:
        # Generate inputs
        logger.info('Generating modelling inputs')
        generate_modelling_inputs(proteins, ligands, models, msa_folder, model_dirs, n_cpus)

        # Generate modelling scripts
        logger.info('Generating modelling scripts')
        modelling_script_paths = generate_modelling_scripts(
            msa_folder,
            models, 
            model_dirs,
            n_predictions,
            logs_foder,
            max_runtime_in_hours=time_budget['modelling'],
        )

    scoring_script_paths = []
    if not skip_scoring:
        # Generate scoring scripts
        logger.info('Generating scoring scripts')
        scoring_script_paths = generate_scoring_scripts()

    # Orchestrate
    logger.info('Generating orchestration script')
    orchestration_script_path = orchestrate_run(
        orchestrator, 
        output_dir,
        msa_script_path, 
        modelling_script_paths, 
        scoring_script_paths,
    )

    # Run
    logger.info('Running orchestration script')
    returncode = do_run(orchestration_script_path)

    if returncode == 0:
        logger.info('DONE')

    sys.exit(returncode)


def generate_msa_script(
    fasta_path : Path, 
    models : List[str], 
    msa_folder : Path,
    logs_foder : Path,
    orchestrator : str,
    max_runtime_in_hours : int,
) -> Path:
    current_path = Path(os.path.abspath(__file__)).parent
    if orchestrator == 'pbspro':
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
            rename_a3m_script=(current_path / 'pbspro' / 'rename_a3m.py').as_posix(),
            protenix_postprocessing_script=(current_path / 'protenix' / 'run_colabfold_postprocess.py').as_posix(),
            chai_postprocessing_script=(current_path / 'chai' / 'run_colabfold_postprocess.py').as_posix(),
        )
    elif orchestrator == 'slurm':
        msa_script_path = current_path / 'af3' / 'run_msa_search.sh'
        with msa_script_path.open('r') as f:
            msa_script_raw = f.read()

        generate_af3_msa_input(fasta_path, msa_folder)

        msa_script = msa_script_raw.format(
            input=msa_folder.resolve().as_posix(),
            output=msa_folder.resolve().as_posix(),
            log_path=(logs_foder / 'af3_msa_run_%j.log').as_posix(),
            time_budget=encode_slurm_time_budget(max_runtime_in_hours),
        )
    else:
        raise ValueError(f'Unknown orchestrator: {orchestrator}')
    
    msa_script_path = (msa_folder / 'run_msa_search.sh')
    with msa_script_path.open('w') as f_out:
        f_out.write(msa_script)

    return msa_script_path


def generate_af3_msa_input(fasta_path : Path, msa_folder : Path) -> None:
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


def generate_modelling_inputs(
    proteins : List[SeqRecord], 
    ligands : pd.DataFrame, 
    models : List[str], 
    msa_folder : Path,
    model_dirs : Dict[str, Path], 
    n_cpus : int,
) -> None:
    for ligand_id, ligand_mol in parse_ligands(ligands, n_cpus):
        for protein in proteins:
            for model in models:
                model_dir = model_dirs[model]

                if model == 'af3':
                    generate_af3_input(protein, ligand_id, ligand_mol, model_dir)
                elif model == 'protenix':
                    generate_protenix_input(protein, ligand_id, ligand_mol, model_dir, msa_folder)
                elif model == 'boltz':
                    generate_boltz_input(protein, ligand_id, ligand_mol, model_dir, msa_folder)
                elif model == 'chai':
                    generate_chai_input(protein, ligand_id, ligand_mol, model_dir, msa_folder)
                else:
                    raise ValueError(f'Unknown model: {model}')


def generate_modelling_scripts(
    msa_folder : Path,
    models : List[str], 
    model_dirs : Dict[str, Path],
    n_predictions : int,
    logs_foder : Path,
    max_runtime_in_hours : int,
):
    script_paths = []
    for model in models:
        model_dir = model_dirs[model]

        if model == 'af3':
            af3_script_path = generate_af3_modelling_script(msa_folder, model_dir, n_predictions, logs_foder, max_runtime_in_hours)
            script_paths.append(af3_script_path)

        if model == 'protenix':
            protenix_script_path = generate_protenix_modelling_script(model_dir, n_predictions, logs_foder, max_runtime_in_hours)
            script_paths.append(protenix_script_path)

        if model == 'boltz':
            boltz_script_path = generate_boltz_modelling_script(model_dir, n_predictions, logs_foder, max_runtime_in_hours)
            script_paths.append(boltz_script_path)
        
        if model == 'chai':
            chai_script_path = generate_chai_modelling_script(model_dir, n_predictions, logs_foder, max_runtime_in_hours)
            script_paths.append(chai_script_path)

    return script_paths


def generate_af3_modelling_script(
    msa_folder : Path, 
    model_dir : Path, 
    n_predictions : int, 
    logs_foder : Path, 
    max_runtime_in_hours : int,
):
    current_path = Path(os.path.abspath(__file__)).parent
    raw_script_path = current_path / 'af3' / 'run_alphafold.sh'
    run_af3_python_script = current_path / 'af3' / 'run_alphafold.py'

    with raw_script_path.open('r') as f:
        af3_script_raw = f.read()

    inputs_dir = model_dir / 'inputs'
    inputs_dir.mkdir(exist_ok=True)

    results_dir = model_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    af3_script = af3_script_raw.format(
        log_path=(logs_foder / 'af3_modelling_%j.log').as_posix(),
        time_budget=encode_slurm_time_budget(max_runtime_in_hours),
        run_af3_python_script=run_af3_python_script.as_posix(),
        protein_specs_dir=msa_folder.resolve().as_posix(),
        ligand_specs_dir=(model_dir / 'ligands').as_posix(),
        af3_inputs_dir=inputs_dir.resolve().as_posix(),
        output_dir=results_dir.resolve().as_posix(),
        n_predictions=n_predictions,
    )

    script_path = (model_dir / 'run_alphafold.sh')
    with script_path.open('w') as f_out:
        f_out.write(af3_script)

    return script_path


def generate_protenix_modelling_script(model_dir : Path, n_predictions : int, logs_foder : Path, max_runtime_in_hours : int):
    current_path = Path(os.path.abspath(__file__)).parent
    raw_script_path = current_path / 'protenix' / 'run_protenix.sh'

    with raw_script_path.open('r') as f:
        protenix_script_raw = f.read()

    results_dir = model_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    protenix_script = protenix_script_raw.format(
        input=(model_dir / 'inputs').resolve().as_posix(),
        output=results_dir.resolve().as_posix(),
        time_budget=encode_pbspro_time_budget(max_runtime_in_hours),
        seeds=gen_protenix_model_seeds(n_predictions),
        log_path=(logs_foder / 'protenix_modelling.log').as_posix(),
    )

    script_path = model_dir / 'run_protenix.sh'
    with script_path.open('w') as f_out:
        f_out.write(protenix_script)

    return script_path


def generate_boltz_modelling_script(model_dir : Path, n_predictions : int, logs_foder : Path, max_runtime_in_hours : int):
    current_path = Path(os.path.abspath(__file__)).parent
    raw_script_path = current_path / 'boltz' / 'run_boltz.sh'

    with raw_script_path.open('r') as f:
        boltz_script_raw = f.read()

    results_dir = model_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    boltz_script = boltz_script_raw.format(
        input=(model_dir / 'inputs').resolve().as_posix(),
        output=results_dir.resolve().as_posix(),
        time_budget=encode_pbspro_time_budget(max_runtime_in_hours),
        seeds=gen_boltz_model_seeds(n_predictions),
        log_path=(logs_foder / 'boltz_modelling.log').as_posix(),
    )

    script_path = model_dir / 'run_boltz.sh'
    with script_path.open('w') as f_out:
        f_out.write(boltz_script)

    return script_path


def generate_chai_modelling_script(model_dir : Path, n_predictions : int, logs_foder : Path, max_runtime_in_hours : int):
    raise NotImplementedError


def generate_scoring_scripts():
    return []


def orchestrate_run(
    orchestrator : str,
    output_dir : Path,
    msa_script_path : Path, 
    modelling_script_paths : List[Path], 
    scoring_script_paths : List[Path],
) -> Path:
    orchestration_script_path = output_dir / 'run_pulldown.sh'

    script_str = (
        '#!/bin/bash\n'
        'set -e\n'
    )
    if orchestrator == 'pbspro':
        msa = False
        if msa_script_path is not None:
            msa = True
            script_str += f'msa_job_id=$(qsub -q hx {msa_script_path.resolve().as_posix()})\n'

        modelling = False
        if len(modelling_script_paths) > 0:
            modelling = True
            for i, modelling_script_path in enumerate(modelling_script_paths):
                if msa:
                    script_str += (
                        f'modelling_job_id_{i}=$(qsub -q hx -W depend=afterok:$msa_job_id {modelling_script_path.resolve().as_posix()})\n'
                    )
                else:
                    script_str += f'modelling_job_id_{i}=$(qsub -q hx {modelling_script_path.resolve().as_posix()})\n'

        if len(scoring_script_paths) > 0:
            for i, scoring_script_path in enumerate(scoring_script_paths):
                if modelling:
                    script_str += f'qsub -q hx -W depend=afterok:$modelling_job_id_{i} {scoring_script_path.resolve().as_posix()}\n'
                else:
                    script_str += f'qsub -q hx {scoring_script_path.resolve().as_posix()}\n'
    
    elif orchestrator == 'slurm':
        msa = False
        if msa_script_path is not None:
            msa = True
            script_str += f'msa_job_id=$(sbatch --parsable {msa_script_path.resolve().as_posix()})\n'

        modelling = False
        if len(modelling_script_paths) > 0:
            modelling = True
            for i, modelling_script_path in enumerate(modelling_script_paths):
                if msa:
                    script_str += (
                        f'modelling_job_id_{i}=$(sbatch --parsable --dependency=afterok:$msa_job_id {modelling_script_path.resolve().as_posix()})\n'
                    )
                else:
                    script_str += f'modelling_job_id_{i}=$(sbatch --parsable {modelling_script_path.resolve().as_posix()})\n'

        if len(scoring_script_paths) > 0:
            for i, scoring_script_path in enumerate(scoring_script_paths):
                if modelling:
                    script_str += f'sbatch --parsable --dependency=afterok:$modelling_job_id_{i} {scoring_script_path.resolve().as_posix()}\n'
                else:
                    script_str += f'sbatch --parsable {scoring_script_path.resolve().as_posix()}\n'
    else:
        raise ValueError(f'Unknown orchestrator: {orchestrator}')

    with orchestration_script_path.open('w') as f_out:
        f_out.write(script_str)

    return orchestration_script_path


def do_run(orchestration_script_path : Path) -> None:
    result = subprocess.run(
        ['bash', orchestration_script_path.resolve().as_posix()],
        stdout=sys.stdout, 
        stderr=sys.stderr,
    )
    return result.returncode


def generate_af3_input(
    protein : SeqRecord, 
    ligand_id : str, 
    ligand_mol : Chem.Mol, 
    model_dir : Path, 
):
    ligand_seq = {
        'ligand': {
            'id': 'B',
            'ccdCodes': [ligand_id],
        }
    }
    try:
        ccd_data = generate_ccd_from_mol(ligand_mol, ligand_id)
    except ValueError:
        logger.error(f'Error for ligand: {ligand_id}')
        raise

    name = f'{protein.id}__{ligand_id}'
    ligand_spec = {
        'name': name,
        'sequences': [ligand_seq],
        'userCCD': ccd_data,
    }

    ligands_dir = model_dir / 'ligands'
    ligands_dir.mkdir(exist_ok=True)

    with (ligands_dir / f'{ligand_id}.json').open('w') as f_out:
        json.dump(ligand_spec, f_out, indent=True)


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

    precomputed_msa_dir = msa_folder / 'protenix_msa' / f'{protein.id}'

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
    name = f'{protein.id}__{ligand_id}'
    spec = {
        'version': 1,
        'sequences': [
            {
                'protein': {
                    'id': ['A'],
                    'sequence': str(protein.seq).upper(),
                    'msa': (msa_folder / f'{protein.id}.a3m').resolve().as_posix(),
                }
            },
            {
                'ligand': {
                    'id': ['B'],
                    'smiles': Chem.MolToSmiles(ligand_mol),
                }
            }
        ],
    }

    output_dir = model_dir / 'inputs'
    output_dir.mkdir(exist_ok=True)

    with (output_dir / f'{name}.yml').open('w') as f_out:
        json.dump(
            spec, 
            f_out,
            indent=True,
        )


def generate_chai_input(
    protein : SeqRecord, 
    ligand_id : str, 
    ligand_mol : Chem.Mol, 
    model_dir : Path, 
    msa_folder : Path,
):
    raise NotImplementedError


def parse_ligands(ligands : pd.DataFrame, n_cpus : int) -> Iterator[Tuple[str, Chem.Mol]]:
    for ligand_id in ligands.index:
        smiles = ligands.loc[ligand_id, 'smiles']
        yield (
            ligand_id,
            generate_conformation(Chem.MolFromSmiles(smiles), n_cpus=n_cpus),
        )


def encode_pbspro_time_budget(max_runtime_in_hours : int) -> str:
    return f'{str(max_runtime_in_hours).zfill(2)}:00:00'


def encode_slurm_time_budget(max_runtime_in_hours : int) -> str:
    days = max_runtime_in_hours // 24
    hours = max_runtime_in_hours - 24 * days
    return f'{days}-{str(hours).zfill(2)}:00:00'


def gen_seeds(n : int, max_seed : int = 1000, n_tries : int = 100) -> List[int]:
    if n >= max_seed:
        raise ValueError(
            f'Number of models requested ({n}) is too high for the max seed set ({max_seed})'
        )

    seeds = []
    for _ in range(n_tries):
        seeds = [int(random.uniform(1, max_seed)) for _ in range(n)]
        if len(seeds) == len(set(seeds)):
            break
    
    if len(seeds) != n:
        # This should never happen
        raise ValueError(f"Couldn't generate {n} unique random seeds")
    
    return seeds


def gen_protenix_model_seeds(n : int, max_seed : int = 1000, n_tries : int = 100) -> str:
    return ','.join([str(s) for s in gen_seeds(n, max_seed, n_tries)])


def gen_boltz_model_seeds(n, max_seed : int = 1000, n_tries : int = 100) -> str:
    return ' '.join([str(s) for s in gen_seeds(n, max_seed, n_tries)])


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
        '--skip_msa', 
        action='store_true',
        help='Skip MSA step',
    )
    parser.add_argument(
        '--skip_modelling', 
        action='store_true',
        help='Skip modelling step',
    )
    parser.add_argument(
        '--skip_scoring', 
        action='store_true',
        help='Skip scoring step',
    )
    parser.add_argument(
        '--n_predictions', 
        type=int,
        default=1,
        help='Number of predictions',
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
