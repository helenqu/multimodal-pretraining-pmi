#!/bin/bash

source env_setup.sh

PAIRS_FILE=$1
IMAGENET_LABEL_OUTDIR=$2
IMAGENET_LABEL_FILE=${IMAGENET_LABEL_OUTDIR}/${basename ${PAIRS_FILE}}
PROMPTS_FILE=data/captions/${basename ${PAIRS_FILE}}
IMAGES_OUTDIR=data/images/${basename ${PAIRS_FILE}}

python src/clip_pretraining/generate_captions/imagenet_label_for_pair.py \
    --pairs_file $PAIRS_FILE \
    --outdir $IMAGENET_LABEL_OUTDIR

accelerate launch --num_processes 1 -m \
    clip_pretraining.generate_captions.generate_captions_dataparallel.py \
    --batch_size 64 \
    --pairs_df $IMAGENET_LABEL_FILE \
    --output_path $PROMPTS_FILE

accelerate launch --num_processes 4 -m \
    clip_pretraining.generate_images.generate_images \
    --batch_size 64 \
    --prompts_file $PROMPTS_FILE \
    --outdir $IMAGES_OUTDIR