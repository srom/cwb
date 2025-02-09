#!/bin/bash
#PBS -l walltime={time_budget}
#PBS -l select=1:ncpus=64:mem=384gb
#PBS -e {error_log_qsub}
#PBS -o {output_log_qsub}

set -e

BASE_DIR="/gpfs/home/rs1521"

INPUT="{input}"
OUTPUT="{output}"
OUTPUT_LOG="{error_log}"
ERROR_LOG="{output_log}"
PROTENIX_POSTPROCESS_SCRIPT="{protenix_postprocessing_script}"
CHAI_POSTPROCESS_SCRIPT="{chai_postprocessing_script}"
run_protenix_postprocessing="{run_protenix_postprocessing}"
run_chai_postprocessing="{run_chai_postprocessing}"

cd $PBS_O_WORKDIR

module purge
module load ColabFold/1.5.2-foss-2022a-CUDA-11.7.0

# Running custom ColabFold version
cd ${BASE_DIR}/ColabFold-1.5.5

python_bin_path="/gpfs/easybuild/prod/software/Python/3.10.4-GCCcore-11.3.0/bin"

echo "Run ColabFold search"
${python_bin_path}/python -m colabfold.mmseqs.search \
	--db1 uniref30_2302_db \
	--db2 pdb100_230517 \
	--db3 colabfold_envdb_202108_db \
	--mmseqs ${BASE_DIR}/MMseqs2/build/bin/mmseqs \
	--threads 64 \
	${INPUT} \
	${BASE_DIR}/colabfold_database/ \
	${OUTPUT} \
	> ${OUTPUT_LOG} \
	2> ${ERROR_LOG}

echo "Rename MSA files"
cd ${BASE_DIR}
module purge
. load_conda.sh
cd ${BASE_DIR}/amp-main

python -m src.db_utils.rename_a3m \
	--a3m_folder ${OUTPUT} \
	--fasta ${INPUT} \
	--output_folder ${OUTPUT} \
	>> ${OUTPUT_LOG} \
	2>> ${ERROR_LOG}

if [[ "$run_protenix_postprocessing" == "true" ]]; then
    echo "Run Protenix postprocessing step"
    module purge
    module load ColabFold/1.5.2-foss-2022a-CUDA-11.7.0
    
    ${python_bin_path}/python ${PROTENIX_POSTPROCESS_SCRIPT} \
	    ${OUTPUT} \
	    >> ${OUTPUT_LOG} \
	    2>> ${ERROR_LOG}
fi

if [[ "$run_chai_postprocessing" == "true" ]]; then
    echo "Run Chai postprocessing step"
	cd ${BASE_DIR}
    module purge
	. load_conda.sh
	conda activate chai

	python ${CHAI_POSTPROCESS_SCRIPT} \
	    ${OUTPUT} \
	    >> ${OUTPUT_LOG} \
	    2>> ${ERROR_LOG}
fi
