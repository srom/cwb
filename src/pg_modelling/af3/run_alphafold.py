import argparse
import copy
import json
import logging
from pathlib import Path
import random
import subprocess
import sys


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser(description='Run AlphaFold 3')
    parser.add_argument(
        '-i', '--protein_specs_dir', 
        type=Path,
        required=True,
        help='Path to directory containing AlphaFold 3 JSON specs for for proteins, with MSA included',
    )
    parser.add_argument(
        '-l', '--ligand_specs_dir', 
        type=Path,
        required=True,
        help='Path to directory containing AlphaFold 3 JSON specs for ligands',
    )
    parser.add_argument(
        '-a', '--af3_inputs_dir', 
        type=Path,
        required=True,
        help='Path to output folder where structures will be saved.',
    )
    parser.add_argument(
        '-o', '--output_dir', 
        type=Path,
        required=True,
        help='Path to output folder where structures will be saved.',
    )
    parser.add_argument(
        '--n_predictions', 
        type=int,
        required=False,
        default=1,
        help='Number of models to predict',
    )
    args = parser.parse_args()

    protein_specs_dir = args.protein_specs_dir
    ligand_specs_dir = args.ligand_specs_dir
    af3_inputs_dir = args.af3_inputs_dir
    output_dir = args.output_dir
    n_predictions = args.n_predictions

    if not protein_specs_dir.is_dir():
        logger.error(f'Protein specs directory path does not exist: {protein_specs_dir}')
        sys.exit(1)
    elif not ligand_specs_dir.is_dir():
        logger.error(f'Ligand specs directory does not exist: {ligand_specs_dir}')
        sys.exit(1)
    elif not af3_inputs_dir.is_dir():
        logger.error(f'AF3 inputs folder does not exist: {af3_inputs_dir}')
        sys.exit(1)
    elif not output_dir.is_dir():
        logger.error(f'Output folder does not exist: {output_dir}')
        sys.exit(1)

    logger.info('Running AlphaFold 3')

    # Generate inputs
    for protein_spec_dir in protein_specs_dir.iterdir():
        if not protein_spec_dir.is_dir():
            continue

        protein_spec_path = protein_spec_dir / f'{protein_spec_dir.name}_data.json'
        with protein_spec_path.open() as f:
            protein_spec = json.load(f)

        for ligand_spec_path in ligand_specs_dir.iterdir():
            if not ligand_spec_path.name.endswith('.json'):
                continue

            with ligand_spec_path.open() as f:
                ligand_spec = json.load(f)

            generate_spec(protein_spec, ligand_spec, n_predictions, af3_inputs_dir)

    # Run AF3
    returncode = run_af3(af3_inputs_dir, output_dir)

    sys.exit(returncode)


def run_af3(specs_dir : Path, output_dir : Path):
    result = subprocess.run(
        [
            'alphafold',
            '--input_dir', specs_dir.resolve().as_posix(),
            '--output_dir', output_dir.resolve().as_posix(),
            '--norun_data_pipeline',
        ],
        stdout=sys.stdout, 
        stderr=sys.stderr,
    )
    return result.returncode


def generate_spec(
    protein_spec : dict,
    ligand_spec : dict,
    n_predictions : int,
    af3_inputs_dir : Path,
):  
    spec = copy.deepcopy(protein_spec)
    name = ligand_spec['name']
    spec['name'] = name
    spec['sequences'] += ligand_spec['sequences']
    spec['userCCD'] = ligand_spec['userCCD']
    spec['modelSeeds'] = gen_model_seeds(n_predictions)

    with (af3_inputs_dir / f'{name}.json').open('w') as f_out:
        json.dump(spec, f_out, indent=True)


def gen_model_seeds(n):
    return [int(random.uniform(1, 1000)) for _ in range(n)]


if __name__ == '__main__':
    main()
