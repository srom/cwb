#!/bin/bash
#SBATCH --job-name=af3_msa_run
#SBATCH --output={log_path}
#SBATCH --partition=gpy
#SBATCH --qos=qos_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time={time_budget}

INPUT="{input}"
INPUT="{output}"

module load alphafold/3.0.1

alphafold \
  --input_dir $INPUT \
  --output_dir $OUTPUT \
  --norun_inference
