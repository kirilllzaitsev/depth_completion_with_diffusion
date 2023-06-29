#!/bin/bash

train_args="--exp_targets only-sr lowres-as-rescaled-256 --num_epochs 2005 --batch_size 2 --max_batch_size 1 --only_super_res --use_super_res --input_res 256 --input_img_size 256 256 --unets_output_res 64 256"

poetry run python train_imagen.py "$train_args"