#!/bin/bash

source env_setup.sh

MODEL_NAME=$1
SAVE_DIR=$2

python -m clip_pretraining.finetune.run \
    --epochs=10  \
    --lr=0.00003  \
    --batch-size=512  \
    --train-dataset=ImageNet \
    --cache-dir=data/cache  \
    --model=$1  \
    --eval-datasets=ImageNetEval \
    --save=$2  \
    --data-location=data/imagenet/val \
    --template=openai \
    --wandb
    # to apply editing transform, uncomment:
    # --apply-editing-transform \
    # --editing-transform-p=0.5 \