#!/bin/bash
#PBS -l walltime={time_budget}
#PBS -l select=1:ncpus=16:mem=128gb:ngpus=1:gpu_type=A100

set -e

n_cpus=16
export OMP_NUM_THREADS=$n_cpus
export MKL_NUM_THREADS=$n_cpus
export OPENBLAS_NUM_THREADS=$n_cpus
export NUMEXPR_NUM_THREADS=$n_cpus
export TBB_NUM_THREADS=$n_cpus
export CPU_COUNT=$n_cpus

cd /gpfs/home/rs1521/

. load_conda.sh
conda activate cwb

cd /gpfs/home/rs1521/cwb-main

python -m src.pg_modelling.protenix.scoring \
    -i "{input}" \
    -o "{output}" \
    > "{log_path}" \
    2>&1

conda deactivate 
conda activate aev-plig

python -m src.pg_modelling.scoring.run_aev_plig_scoring \
    -i "{output}" \
    >> "{log_path}" \
    2>&1
