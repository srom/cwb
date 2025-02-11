#!/bin/bash
#PBS -l walltime={time_budget}
#PBS -l select=1:ncpus=8:mem=384gb:ngpus=1:gpu_type=A100

cd /gpfs/home/rs1521/

. load_conda.sh
conda activate boltz

input="{input}"
output="{output}"
log_path="{log_path}"
seeds=({seeds})

for seed in "${{seeds[@]}}"; do
    out_dir="$output/seed_$seed"
    mkdir $out_dir

    boltz predict \
        $input \
        --out_dir $out_dir \
        --cache boltz_data \
        --accelerator gpu \
        --diffusion_samples 5 \
        --seed $seed \
        >> $log_path \
        2>>&1
done
