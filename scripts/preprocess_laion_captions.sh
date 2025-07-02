#!/bin/bash

source env_setup.sh

ROOT_DIR=$1

for file in $ROOT_DIR/*.parquet; do
    # tokenize laion captions
    python src/clip_pretraining/build_pair_to_freq/tokenize_laion.py \
        --input "$file"
    
    # find and count pairs
    cleaned_file="$(dirname "$file")/cleaned/$(basename "${file%.*}")_cleaned.snappy.parquet"
    python src/clip_pretraining/build_pair_to_freq/build_pair_freqs.py \
        --input "$cleaned_file"
    
    # write zero frequency pairs (pairings of laion caption words that aren't paired)
    python src/clip_pretraining/build_pair_to_freq/write_freq0_pairs_for_file.py \
        --input "$cleaned_file"
done

# filter all pairs by imagenet words
for file in $ROOT_DIR/cleaned/*pair_freqs.snappy.parquet; do
    python src/clip_pretraining/build_pair_to_freq/filter_pairs.py \
        --input "$file" \
        --imagenet
done

for file in $ROOT_DIR/cleaned/*_freq0-pairs.gz; do
    python src/clip_pretraining/build_pair_to_freq/filter_pairs.py \
        --input "$file" \
        --imagenet
done

# combine pairs from all shards into a single file
python src/clip_pretraining/build_pair_to_freq/combine_filtered_freq0_pairs.py
python src/clip_pretraining/build_pair_to_freq/combine_filtered_nonzero_freq_pairs.py