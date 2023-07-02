#!/bin/bash

cd /path/to/PENet_ICRA2021 || exit 1

result_dir=results/custom

mkdir -p $result_dir

python main.py -b 1 -n pe --evaluate models/pe.pth.tar --data-folder-save $result_dir --test --cpu --save_prediction --dataset-paths-dir /path/to/test/paths