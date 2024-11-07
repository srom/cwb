#!/bin/bash

#####################################################################################
# This script makes a phylogenetic tree from amino acid sequence.
#
# The following software need to be available in $PATH:
# - MAFFT:    https://mafft.cbrc.jp/alignment/software/
# - trimAl:   http://trimal.cgenomics.org/
# - IQ-TREE:  http://www.iqtree.org/
#
######################################################################################

set -e
start=$(date +%s%N)

source tree_helper.sh

##
### Arguments
##

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --work_dir) work_dir="$2"; shift ;;
        --fasta) fasta="$2"; shift ;;
        --prefix) prefix="$2"; shift ;;
        --n_cpus) n_cpus="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${work_dir}" ] || [ ! -d "${work_dir}" ]; then
    echo "--work_dir is not set or not a directory"
    exit 1
fi

if [ ! -z "${fasta}" ] && [ ! -f "${fasta}" ]; then
    echo "--fasta is not a file"
    exit 1
fi

if [ -z "${prefix}" ]; then
    prefix="tree_aa"
fi

if [ -z "${n_cpus}" ]; then
    echo "Warning: --n_cpus is not set. Using default value of 4"
    n_cpus=4
fi

##
### Script
##

input_fasta="${work_dir}/sequences.fasta"
alignment_fasta="${work_dir}/sequences.aln.fasta"
alignment_trimmed_fasta="${work_dir}/sequences.aln.trimmed.fasta"
final_fasta="${work_dir}/alignment_final.fasta"

cp ${fasta} ${input_fasta}

##
# Align protein sequences with MAFFT-LINSI.
##

echo "Align protein sequences with MAFFT (mafft-linsi)"
mafft_start=$(date +%s%N)

mafft-linsi \
    --reorder \
    --thread ${n_cpus} \
    --threadtb $(($n_cpus / 2)) \
    --threadit $(($n_cpus / 2)) \
    ${input_fasta} \
    > ${alignment_fasta}

mafft_end=$(date +%s%N)
echo "Alignment elapsed time: $(calculate_elapsed_time $mafft_start $mafft_end)"

##
# Trim alignment positions with less than 35% data.
##
echo "Trim alignment with less than 35% data"
trimal_start=$(date +%s%N)
trimal -in ${alignment_fasta} -out ${alignment_trimmed_fasta} -gt 0.35
trimal_end=$(date +%s%N)
echo "trimAl elapsed time: $(calculate_elapsed_time $trimal_start $trimal_end)"

##
# Remove sequences with more than 50% gaps.
##
remove_sequences_with_too_many_gaps ${alignment_trimmed_fasta} ${final_fasta} 0.5

##
# Run IQ-TREE.
##
echo "Run IQ-TREE"
iqtree_start=$(date +%s%N)

pushd "${work_dir}" > /dev/null

iqtree \
    -s ${final_fasta} \
    -pre ${prefix} \
    -m MFP \
    -B 1000 \
    -alrt 1000 \
    -bnni \
    -T AUTO \
    --threads-max ${n_cpus}

popd > /dev/null

iqtree_end=$(date +%s%N)
echo "IQ-TREE elapsed time: $(calculate_elapsed_time $iqtree_start $iqtree_end)"

end=$(date +%s%N)
echo "Total elapsed time: $(calculate_elapsed_time $start $end)"
