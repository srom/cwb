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

alphafold \
    --input_dir {input} \
    --output_dir {output} \
    --norun_data_pipeline
