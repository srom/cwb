"""
Script to run a pulldown using one protein (or protein complex) as bait against a library of ligands.

This script assumes that the MSA part of the pipeline has been run already (i.e. AF3 run with flag --norun_inference)
Hence, a JSON file in AF3 format must already be available.

AlphaFold 3 (command `alphafold`) is assumed to be available in $PATH.

Arguments:
- Path to AF3 compatible JSON spec for proteins with MSAs.
- Path to directory containing AF3 compatible JSON specs for ligands.
- Number of models to be predicted.
- Output folder.

Generate a new JSON spec per ligand and runs AF3.

NOTE: a way to specify MSA as external files is available in the input template version 2. Not released yet.
TODO: use template version 2 when released.
"""
import argparse
import copy
import json
import logging
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser(description='AF3 ligand pulldown')
    parser.add_argument(
        '-i', '--protein_spec_path', 
        type=Path,
        required=True,
        help='Path to AlphaFold 3 JSON spec for proteins with MSA',
    )
    parser.add_argument(
        '-l', '--ligand_specs_dir', 
        type=Path,
        required=True,
        help='Path to directory containing AlphaFold 3 JSON specs for ligands',
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
    parser.add_argument(
        '--batch_size', 
        type=int,
        required=False,
        default=10,
        help='AF3 seem to leak memory / store all inputs in memory; restart every so often.',
    )
    args = parser.parse_args()

    protein_spec_path = args.protein_spec_path
    ligand_specs_dir = args.ligand_specs_dir
    output_folder = args.output_folder
    n_models = args.n_models
    batch_size = args.batch_size

    if not protein_spec_path.is_file():
        logger.error(f'Spec path does not exist: {protein_spec_path}')
        sys.exit(1)
    elif not ligand_specs_dir.is_dir():
        logger.error(f'Ligands directory does not exist: {ligand_specs_dir}')
        sys.exit(1)
    elif not output_folder.is_dir():
        logger.error(f'Output folder does not exist: {output_folder}')
        sys.exit(1)

    logger.info('Loading protein spec')
    with protein_spec_path.open() as f:
        protein_spec = json.load(f)

    logger.info('Loading ligand specs')
    ligand_specs = []
    for p in ligand_specs_dir.iterdir():
        if p.name.endswith('.json'):
            with p.open() as f:
                ligand_spec = json.load(f)
                ligand_specs.append(ligand_spec)

    returncode = 0
    n = len(ligand_specs)
    n_iterations = math.ceil(n / batch_size)
    for i in range(n_iterations):
        start = i * batch_size
        end = min(start + batch_size, n)

        logger.info(f'Batch: {start+1:,} to {end:,} of {n:,}')

        tempdir = Path(tempfile.mkdtemp())
        try:
            save_json_specs(protein_spec, ligand_specs[start:end], tempdir, n_models)
            returncode = run_af3(tempdir, output_folder)

            if returncode != 0:
                sys.exit(returncode)
        finally:
            shutil.rmtree(tempdir)

    if returncode == 0:
        logger.info('DONE')
    else:
        logger.error('DONE with errors')

    sys.exit(returncode)


def save_json_specs(protein_spec, ligands_specs, specs_dir, n_models):
    for ligands_spec in ligands_specs:
        ligand_id = ligands_spec['name']
        name = protein_spec['name'] + f'__{ligand_id}'
        spec = copy.deepcopy(protein_spec)
        spec['name'] = name
        spec['modelSeeds'] = gen_model_seeds(n_models)

        spec['sequences'] += ligands_spec['sequences']

        if 'userCCD' in ligands_spec:
            spec['userCCD'] = ligands_spec['userCCD']

        with (specs_dir / f'{name}.json').open('w') as f_out:
            json.dump(spec, f_out, indent=True)


def run_af3(specs_dir, output_folder):
    result = subprocess.run(
        [
            'alphafold',
            '--input_dir', specs_dir.resolve().as_posix(),
            '--output_dir', output_folder.resolve().as_posix(),
            '--norun_data_pipeline',
        ],
        stdout=sys.stdout, 
        stderr=sys.stderr,
    )
    return result.returncode


def gen_model_seeds(n):
    return [int(random.uniform(1, 100)) for _ in range(n)]


if __name__ == '__main__':
    main()
