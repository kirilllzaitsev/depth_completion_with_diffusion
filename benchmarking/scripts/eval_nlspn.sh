#!/bin/bash

nlspn_dir=/path/to/NLSPN_ECCV20
cd $nlspn_dir || exit 1

python src/main.py --dir_data /media/master/text/cv_data/code_that_needs_cv_data/nlspn/nlspn --data_name KITTIDC --split_json $nlspn_dir/data_json/kitti_dc_test.json --patch_height 240 --patch_width 1216 --gpus 0 --max_depth 90.0 --num_sample 0 --test_only --pretrain $nlspn_dir/models/NLSPN_KITTI_DC.pt --save test --legacy --network resnet18 --batch_size 1  --dataset-paths-dir /path/to/val/paths