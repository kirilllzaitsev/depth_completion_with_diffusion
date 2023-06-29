#!/bin/bash

cd /media/master/wext/msc_studies/second_semester/research_project/related_work/NLSPN_ECCV20 || exit 1

python src/main.py --dir_data /media/master/text/cv_data/code_that_needs_cv_data/nlspn/nlspn --data_name KITTIDC --split_json /media/master/wext/msc_studies/second_semester/research_project/related_work/NLSPN_ECCV20/data_json/kitti_dc_test.json --patch_height 240 --patch_width 1216 --gpus 0 --max_depth 90.0 --num_sample 0 --test_only --pretrain /media/master/wext/msc_studies/second_semester/research_project/related_work/NLSPN_ECCV20/models/NLSPN_KITTI_DC.pt --save test --legacy --network resnet18 --batch_size 1