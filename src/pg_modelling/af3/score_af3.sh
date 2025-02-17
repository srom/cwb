#!/bin/bash
#SBATCH --job-name=af3_scoring
#SBATCH --output={log_path}
#SBATCH --partition=gpu
#SBATCH --qos=qos_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time={time_budget}

set -e

n_cpus=16
export OMP_NUM_THREADS=$n_cpus
export MKL_NUM_THREADS=$n_cpus
export OPENBLAS_NUM_THREADS=$n_cpus
export NUMEXPR_NUM_THREADS=$n_cpus
export TBB_NUM_THREADS=$n_cpus
export CPU_COUNT=$n_cpus

module load miniconda3
conda activate cwb

cd /home/rs1521/cwb-main

python -m src.pg_modelling.af3.scoring \
    -i "{input}" \
    -o "{output}"

conda deactivate 
conda activate aev-plig

python -m src.pg_modelling.scoring.run_aev_plig_scoring -i "{output}"
