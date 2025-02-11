#!/bin/bash
#PBS -l walltime={time_budget}
#PBS -l select=1:ncpus=8:mem=384gb:ngpus=1:gpu_type=A100

set -e

cd /gpfs/home/rs1521/

. load_conda.sh
conda activate chai

input="{input}"
output="{output}"
msa_folder="{msa_folder}"
log_path="{log_path}"
seeds=({seeds})

find "$input" -type f -name "*.fasta" | while read input_fasta; do
    echo "Processing $input_fasta"
    filename=$(basename "$input_fasta" .fasta)
    intermediate_dir="$output/$filename"
    mkdir "$intermediate_dir"

    for seed in "${{seeds[@]}}"; do
        out_dir="$intermediate_dir/seed_$seed"
        mkdir "$out_dir"

        chai fold \
            "$input_fasta" \
            "$out_dir" \
            --msa-directory "$msa_folder" \
            --seed $seed \
            >> "$log_path" \
            2>&1
    done
done
