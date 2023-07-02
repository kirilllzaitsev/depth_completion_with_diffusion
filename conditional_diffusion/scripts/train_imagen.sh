#!/bin/bash

train_args_only_sr="--exp_targets only-sr lowres-as-rescaled-256 --num_epochs 2005 --batch_size 2 --max_batch_size 1 --only_super_res --use_super_res --input_res 256 --input_img_size 256 256 --unets_output_res 64 256"
train_args_only_base="--logdir-name debug --num_epochs 1200 --only_base --input_res 64 --input_img_size 64 64 --unets_output_res 64 --exp_targets debug-grads"

# train_args=$train_args_only_base
train_args=$train_args_only_sr

poetry run python train_imagen.py "$train_args"