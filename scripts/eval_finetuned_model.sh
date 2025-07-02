#!/bin/bash

source env_setup.sh

MODEL_PATH=$1
DATA_DIR=$2

python -m \
  clip_pretraining.finetune.eval \
  --model_path $MODEL_PATH/finetuned/checkpoint_10.pt \
  --eval-datasets ImageNetEval \
  --data-location $DATA_DIR