#!/bin/bash
#PBS -l walltime={time_budget}
#PBS -l select=1:ncpus=8:mem=384gb:ngpus=1:gpu_type=A100

cd /gpfs/home/rs1521/

. load_conda.sh
conda activate protenix

protenix predict \
    --input "{input}" \
    --out_dir "{output}" \
    --seeds "{seeds}" \
    > "{log_path}" \
    2>&1
