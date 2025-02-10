#!/bin/bash
#SBATCH --job-name=af3_msa_run
#SBATCH --output={log_path}
#SBATCH --partition=cpu
#SBATCH --qos=qos_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={time_budget}

set -e

export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""

INPUT="{input}"
INPUT="{output}"

module load alphafold/3.0.0

alphafold \
  --input_dir $INPUT \
  --output_dir $OUTPUT \
  --norun_inference
