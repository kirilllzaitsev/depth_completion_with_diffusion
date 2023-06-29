# Self-supervised depth completion

## Contents

The project consists of two independent parts:

- conditional_diffusion - implementation of the conditional diffusion model from the semester thesis.
- rsl_depth_completion - training pipeline for self-supervised depth completion

## Installation

### Pre-requisites

#### Software

- Python >=3.7
- Poetry ~=1.4.0

```bash
python --version
curl -sSL https://install.python-poetry.org | python -
```

#### Data

- Download the KITTI Depth Completion dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) to a PATH_TO_DEPTH_COMPLETION directory of your choice.
- Replace '???' with PATH_TO_DEPTH_COMPLETION in all files under the rsl_depth_completion/configs/dataset directory.

Expected structure of the dataset directory:

```bash
/media/master/text/cv_data/kitti-full
├── data
│   ├── kitti_depth_completion
│   ├── kitti_depth_completion_kbnet
│   ├── kitti_raw_data
│   ├── train_val_test_file_paths_kbnet
│   ├── train_val_test_file_paths_raw
│   └── train_val_test_file_paths_raw.zip
├── kitti-calibration
│   └── calib_cam_to_cam.txt
└── nlspn

nlspn
├── depth_selection -> /media/master/text/cv_data/kitti-depth-completion/data_depth_selection
├── train
│   ├── 2011_09_26_drive_0001_sync
│       ├── calib_cam_to_cam.txt
│       ├── calib_imu_to_velo.txt
│       ├── calib_velo_to_cam.txt
│       ├── image_02 -> /media/master/text/cv_data/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02
│       ├── image_03 -> /media/master/text/cv_data/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_03
│       ├── oxts -> /media/master/text/cv_data/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/oxts
│       └── proj_depth
└── val
    ├── 2011_09_26_drive_0002_sync
       ├── calib_cam_to_cam.txt
       ├── calib_imu_to_velo.txt
       ├── calib_velo_to_cam.txt
       ├── image_02 -> /media/master/text/cv_data/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0002_sync/image_02
       ├── image_03 -> /media/master/text/cv_data/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0002_sync/image_03
       ├── oxts -> /media/master/text/cv_data/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0002_sync/oxts
       └── proj_depth
```

depth_completion_symlinks.ipynb can be used to create symlinks to duplicated parts of the dataset from above
nlspn_prepare_KITTI_DC.py creates symlinks to match NLSPN format

Before executing the above scripts, create a meta.py file with the following content, replacing placeholders with the correct paths:

```python
import os

if os.path.exists("/cluster"):
    base_dir = "/cluster/scratch/path/to/kitti_dataset"
    base_dir_depth_completion = "/cluster/scratch/path/to/kitti_depth_completion"
else:
    base_dir = "/local/path/to/kitti_dataset"
    base_dir_depth_completion = "/local/path/to/kitti_depth_completion"
```

### Python dependencies

```bash
poetry install
```

### Submodules

```bash
git submodule update --init --recursive
cd submodules || exit 1
for d in */ ; do (cd "$d" && pip install --no-deps -e .); done
```

## Usage

For the conditional diffusion model for depth completion, see the README in the conditional_diffusion directory: [conditional_diffusion/README.md](conditional_diffusion/README.md).

For the self-supervised depth completion pipeline, see the README in the rsl_depth_completion directory: [rsl_depth_completion/README.md](rsl_depth_completion/README.md).

## Benchmarking

Considered models:

- [NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20)
- [PENet](https://github.com/JUGGHM/PENet_ICRA2021)
- [KBNet](https://github.com/alexklwong/calibrated-backprojection-network)

Before running the models, make sure to install their dependencies in separate Python environments. Also, download the pretrained models from the above links and place them in the corresponding directories (see the READMEs of the models for details).

To run the models, consider cloning forks instead of the original repositories, as they contain some fixes to make them easier to benchmark (the core logic is unchanged). Forked repositories:

- [NLSPN](https://github.com/kirilllzaitsev/NLSPN_ECCV20)
- [PENet](https://github.com/kirilllzaitsev/PENet_ICRA2021)
- [KBNet](https://github.com/kirilllzaitsev/calibrated-backprojection-network)

Note. NLSPN must be run on Euler, while KBNet and PENet can fit on 8Gb GPU.

### Tool to analyze benchmark results

## Acknowledgements

- <https://github.com/fangchangma/self-supervised-depth-completion> - base code for data loading and self-supervised losses.
- <https://github.com/alexklwong/calibrated-backprojection-network> - base code for additional loss functions.
- <https://github.com/ashleve/lightning-hydra-template#best-practices> - template for the project structure.
