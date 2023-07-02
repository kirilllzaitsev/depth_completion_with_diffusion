#!/bin/bash

train_args="--logdir-name final_results --use-ssl --lr 1e-5 --num_epochs 25 --only_super_res --input_res 256 --input_img_size 256 256 --unets_output_res 64 256 --exp_targets ssl --sampling_freq 1 --max_batch_size 1 --batch_size 4 --trainer_ckpt_path 'models/plain_centerline_7375-unet-2-last.pt' --config-path configs/full_dataset.yaml"

poetry run python train_imagen.py "$train_args"