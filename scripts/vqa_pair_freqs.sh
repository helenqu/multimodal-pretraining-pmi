#!/bin/bash

source env_setup.sh

for file in $(ls data/cleaned/*pair_freqs.snappy.parquet); do
    python vlm_eval/vqa_pair_freqs_job.py --freqs_path $file --pairs_path $1
done