#!/bin/bash

source env_setup.sh

python -m clip_pretraining.imagenet_pairs.autogenerate_accessory_images

python -m clip_pretraining.imagenet_paste.generate_imagenetpaste_eval \
    --seed 12345