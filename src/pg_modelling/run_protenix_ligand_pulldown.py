"""
Script to run a pulldown using a protein as bait against a library of ligands using [Protenix](https://github.com/bytedance/Protenix).

This script assumes that Mulltiple Sequence Alignments (MSA) are available for the protein (i.e. generated with ColabFold + Protenix postprocessing).

Protenix (command `protenix`) is assumed to be available in $PATH.

Arguments:
- Path to fasta file containing the protein sequence.
- Path to MSA folder containing paired and unpaired MSA files.
- Number of models to be predicted.
- Output folder.

Generates an appropriate JSON spec and runs Protenix.

Limitations: only handles a single protein at the moment; can easily be extended to more complex cases.
"""
import argparse
import json
import logging
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

from Bio import SeqIO, SeqRecord


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser(description='Protenix ligand pulldown')
    parser.add_argument(
        '-i', '--protein_fasta_path', 
        type=Path,
        required=True,
        help='Path to protein fasta file',
    )
    parser.add_argument(
        '-m', '--msa_folder', 
        type=Path,
        required=True,
        help='Path to folder containing MSA files',
    )
    parser.add_argument(
        '-l', '--ligand_pdb_dir', 
        type=Path,
        required=True,
        help='Path to directory containing ligands in PDB format',
    )
    parser.add_argument(
        '-o', '--output_folder', 
        type=Path,
        required=True,
        help='Path to output folder where structures will be saved.',
    )
    parser.add_argument(
        '--n_models', 
        type=int,
        required=False,
        default=1,
        help='Generate that many models for each input',
    )
    args = parser.parse_args()

    protein_fasta_path = args.protein_fasta_path
    msa_folder = args.msa_folder
    ligand_pdb_dir = args.ligand_pdb_dir
    output_folder = args.output_folder
    n_models = args.n_models

    if not protein_fasta_path.is_file():
        logger.error(f'Spec path does not exist: {protein_fasta_path}')
        sys.exit(1)
    elif not msa_folder.is_dir():
        logger.error(f'MSA directory does not exist: {msa_folder}')
        sys.exit(1)
    elif not ligand_pdb_dir.is_dir():
        logger.error(f'Ligands directory does not exist: {ligand_pdb_dir}')
        sys.exit(1)
    elif not output_folder.is_dir():
        logger.error(f'Output folder does not exist: {output_folder}')
        sys.exit(1)

    protein_records = list(SeqIO.parse(protein_fasta_path, 'fasta'))
    if len(protein_records) == 0:
        logger.error(f'No sequences found in the FASTA file: {protein_fasta_path}')
        sys.exit(1)
    elif len(protein_records) > 1:
        logger.warning('More than one sequence in the FASTA file; only the first sequence will be considered.')

    protein_record = protein_records[0]

    tempdir = Path(tempfile.mkdtemp())
    try:
        json_path = save_json_specs(protein_record, msa_folder, ligand_pdb_dir, tempdir)
        returncode = run_protenix(json_path, output_folder, n_models)
    finally:
        shutil.rmtree(tempdir)

    if returncode == 0:
        logger.info('DONE')
    else:
        logger.error('DONE with errors')

    sys.exit(returncode)


def save_json_specs(protein_record : SeqRecord, msa_folder : Path, ligand_pdb_dir : Path, tempdir : Path) -> Path:
    data = []
    for ligand_pdb_path in ligand_pdb_dir.glob('*.pdb'):
        ligand_name = ligand_pdb_path.name.replace('.pdb', '')
        name = f'{protein_record.id}__{ligand_name}'

        sequences = [
            {
                'proteinChain': {
                    'sequence': str(protein_record.seq),
                    'count': 1,
                    'msa': {
                        'precomputed_msa_dir': msa_folder.resolve().as_posix(),
                        'pairing_db': 'uniref100'
                    }
                }
            },
            {
                'ligand': {
                    'ligand': f'FILE_{ligand_pdb_path.resolve().as_posix()}',
                    'count': 1,
                }
            }
        ]
        spec = {
            'name': name,
            'sequences': sequences,
        }
        data.append(spec)

    if len(data) == 0:
        raise ValueError(f'No PDB file found in {ligand_pdb_dir}')

    output_path = tempdir / f'{protein_record.id}_pulldown_inputs.json'
    with output_path.open('w') as f_out:
        json.dump(
            data, 
            f_out,
            indent=True,
        )

    return output_path


def run_protenix(json_path : Path, output_folder : Path, n_models : int) -> int:
    result = subprocess.run(
        [
            'protenix', 'predict',
            '--input', json_path.resolve().as_posix(),
            '--out_dir', output_folder.resolve().as_posix(),
            '--seeds', gen_model_seeds(n_models),
        ],
        stdout=sys.stdout, 
        stderr=sys.stderr,
    )
    return result.returncode


def gen_model_seeds(n : int, max_seed : int = 1000, n_tries : int = 100) -> str:
    if n >= max_seed:
        raise ValueError(
            f'Number of models requested ({n}) is too high for the max seed set ({max_seed})'
        )

    for _ in range(n_tries):
        seeds = [int(random.uniform(1, max_seed)) for _ in range(n)]
        if len(seeds) == len(set(seeds)):
            break
    
    if len(seeds) != n:
        # This should never happen
        raise ValueError(f"Couldn't generate {n} unique random seeds")

    return ','.join([str(s) for s in seeds])


if __name__ == '__main__':
    main()
