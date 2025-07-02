#!/bin/bash

source env_setup.sh

IMAGES_DIR=$1
MODEL=$2

python -m clip_pretraining.zero_shot_imagenet.run_zero_shot \
    --data_dir $IMAGES_DIR \
    --model $MODEL