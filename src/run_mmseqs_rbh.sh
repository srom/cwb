#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=64:mem=64gb
#PBS -e error_run_mmseqs_rbh.txt
#PBS -o output_run_mmseqs_rbh.txt

cd /rds/general/user/rs1521/home

. load_conda.sh
conda activate mmseqs

INPUT="/rds/general/user/rs1521/home/Streptococcus/Streptococcus_all_proteins.fasta"
OUTPUT="/rds/general/user/rs1521/home/Streptococcus/Streptococcus_orthologs.tsv"
USER_TMP="/rds/general/ephemeral/user/rs1521/ephemeral/"

mmseqs easy-rbh \
	${INPUT} \
	${INPUT} \
	${OUTPUT} \
	${USER_TMP} \
	--threads 64
