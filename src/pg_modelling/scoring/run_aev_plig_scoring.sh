#!/bin/bash

set -e

dataset_csv="$1"
data_name="$2"

pushd ../AEV-PLIG-main

python process_and_predict.py \
    --dataset_csv="$dataset_csv" \
    --data_name="$data_name" \
    --trained_model_name=model_GATv2Net_ligsim90_fep_benchmark

popd
