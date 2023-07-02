#!/bin/bash

cd /path/to/PENet_ICRA2021 || exit 1

python main.py -b 1 -n pe --evaluate models/pe.pth.tar --dataset-paths-dir /path/to/val/paths