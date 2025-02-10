#! /bin/bash

#SBATCH --job-name=af3_modelling
#SBATCH --output={log_path}
#SBATCH --partition=gpu
#SBATCH --qos=qos_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time={time_budget}

module load alphafold/3.0.1

python {run_af3_python_script} \
    --protein_specs_dir "{protein_specs_dir}" \
    --ligand_specs_dir "{ligand_specs_dir}" \
    --af3_inputs_dir "{af3_inputs_dir}" \
    --output_dir "{output_dir}" \
    --n_predictions {n_predictions}
