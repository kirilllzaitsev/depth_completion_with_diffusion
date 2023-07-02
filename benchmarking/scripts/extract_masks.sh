#!/bin/bash

cd Mask2Former || exit 1

out_dir=demo/results/masks

mkdir -p $out_dir

python demo/extract_masks.py --path-to-data-config demo/kitti_val_image.txt --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml --output $out_dir --opts MODEL.WEIGHTS models/weights/mask2former-cityscapes-segm-swins.pkl