#!/bin/bash

module load gcc/6.3.0 python_gpu/3.8.5
source /cluster/home/kzaitse/venvs/nlspn/bin/activate

export NLSPN_ROOT=/cluster/home/kzaitse/benchmarking/NLSPN_ECCV20
cd $NLSPN_ROOT || exit 1

# export PATH_TO_NYUv2=
export PATH_TO_KITTI_DC=/cluster/scratch/kzaitse/extracted/nlspn
export PATH_TO_PRETRAINED_MODEL_DIR="$NLSPN_ROOT/models/NLSPN_KITTI_DC.pt"
export PATH_TO_WEIGHTS="$NLSPN_ROOT/models/NLSPN_KITTI_DC.pt"
export NAME_TO_SAVE="custom_fixed"
export SPLIT_JSON="$NLSPN_ROOT/data_json/kitti_dc_val.json"

# An example command for NYUv2 dataset testing
# python main.py --dir_data $PATH_TO_NYUv2 --data_name NYU  --split_json ../data_json/nyu.json \
#     --patch_height 228 --patch_width 304 --gpus 0 --max_depth 10.0 --num_sample 500 \
#     --test_only --pretrain ../results/NLSPN_NYU.pt --preserve_input --save $NAME_TO_SAVE --legacy

# An example command for KITTI DC dataset testing
# python main.py --dir_data $PATH_TO_KITTI_DC --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
#     --patch_height 240 --patch_width 1216 --gpus 0 --max_depth 90.0 --num_sample 0 \
#     --test_only --pretrain ../results/NLSPN_KITTI_DC.pt --preserve_input --save $NAME_TO_SAVE --legacy \
# --batch_size 1 --network resnet18

# An example command for KITTI DC Online evaluation data generation
python src/main.py --dir_data $PATH_TO_KITTI_DC --data_name CustomKITTIDC --split_json $SPLIT_JSON \
    --patch_height 240 --patch_width 1216 --gpus 0 --max_depth 90.0 --num_sample 0 \
    --test_only --pretrain $PATH_TO_PRETRAINED_MODEL_DIR --preserve_input --save_image --save_result_only \
    --save $NAME_TO_SAVE --legacy --batch_size 1 --network resnet34