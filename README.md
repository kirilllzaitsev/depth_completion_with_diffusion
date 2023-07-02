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

**Before starting**, make sure to update all paths that start with '/path/to' in the directory you want to work with.

- Download the KITTI Depth Completion dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) to a PATH_TO_DEPTH_COMPLETION directory of your choice.
- Replace '???' with PATH_TO_DEPTH_COMPLETION in all files under the rsl_depth_completion/configs/dataset directory.

Expected structure of the dataset directory:

```bash
/path/to/kitti-full
├── data
│   ├── kitti_depth_completion <- data from the KITTI Depth Completion dataset
│   │    ├── testing
│   │    │   ├── image
│   │    │   ├── intrinsics
│   │    │   └── sparse_depth
│   │    ├── train_val_split
│   │    │   ├── ground_truth
│   │    │   └── sparse_depth
│   │    └── validation
│   │        ├── ground_truth
│   │        ├── image
│   │        ├── intrinsics
│   │        └── sparse_depth
│   ├── kitti_depth_completion_kbnet <- data for KBNet (see the section on data preparation in the KBNet's README.md)
│        ├── data
│        │   ├── 2011_09_26
│        │   ├── 2011_09_28
│        │   ├── 2011_09_29
│        │   ├── 2011_09_30
│        │   └── 2011_10_03
│        ├── testing
│        │   ├── image
│        │   ├── intrinsics
│        │   └── validity_map
│        ├── train_val_split
│        │   ├── image
│        │   └── validity_map
│        └── validation
│            ├── image
│            ├── intrinsics
│            └── validity_map
│   ├── kitti_raw_data <- raw data from KITTI
│        ├── 2011_X
│        └── 2011_10_03
│            ├── 2011_10_03_drive_0027_sync
│            │   ├── image_00
│            │   ├── image_01
│            │   ├── image_02
│            │   ├── image_03
│            │   ├── oxts
│            │   └── velodyne_points
│            ├── 2011_10_03_drive_Y_sync
│            │   ├── image_00
│            │   ├── image_01
│            │   ├── image_02
│            │   ├── image_03
│            │   ├── oxts
│            │   └── velodyne_points
│            ├── calib_cam_to_cam.txt
│            ├── calib_imu_to_velo.txt
│            └── calib_velo_to_cam.txt
│   ├── train_val_test_file_paths_kbnet <- paths to data used for training, validation and testing of KBNet (see the section on data preparation in the KBNet's README.md)
│        ├── testing
│        │   ├── kitti_test_image.txt
│        │   ├── kitti_test_intrinsics.txt
│        │   ├── kitti_test_sparse_depth.txt
│        │   └── kitti_test_validity_map.txt
│        ├── training
│        │   ├── kitti_train_ground_truth-clean.txt
│        │   ├── kitti_train_ground_truth.txt
│        │   ├── kitti_train_image-clean.txt
│        │   ├── kitti_train_image.txt
│        │   ├── kitti_train_intrinsics-clean.txt
│        │   ├── kitti_train_intrinsics.txt
│        │   ├── kitti_train_sparse_depth-clean.txt
│        │   ├── kitti_train_sparse_depth.txt
│        │   ├── kitti_train_validity_map-clean.txt
│        │   ├── kitti_train_validity_map.txt
│        │   ├── kitti_unused_ground_truth.txt
│        │   ├── kitti_unused_image.txt
│        │   ├── kitti_unused_intrinsics_depth.txt
│        │   ├── kitti_unused_sparse_depth.txt
│        │   └── kitti_unused_validity_map.txt
│        ├── training_trunc
│        │   ├── kitti_train_ground_truth-clean.txt
│        │   ├── kitti_train_ground_truth.txt
│        │   ├── kitti_train_image-clean.txt
│        │   ├── kitti_train_image.txt
│        │   ├── kitti_train_intrinsics-clean.txt
│        │   ├── kitti_train_intrinsics.txt
│        │   ├── kitti_train_sparse_depth-clean.txt
│        │   ├── kitti_train_sparse_depth.txt
│        │   ├── kitti_train_validity_map-clean.txt
│        │   ├── kitti_train_validity_map.txt
│        │   ├── kitti_unused_ground_truth.txt
│        │   ├── kitti_unused_image.txt
│        │   ├── kitti_unused_intrinsics_depth.txt
│        │   ├── kitti_unused_sparse_depth.txt
│        │   └── kitti_unused_validity_map.txt
│        └── validation
│            ├── kitti_val_ground_truth.txt
│            ├── kitti_val_image.txt
│            ├── kitti_val_intrinsics.txt
│            ├── kitti_val_sparse_depth.txt
│            └── kitti_val_validity_map.txt
│   ├── train_val_test_file_paths_raw <- paths to raw data as downloaded from the KITTI website
│        ├── testing
│        │   ├── kitti_test_image.txt
│        │   ├── kitti_test_intrinsics.txt
│        │   ├── kitti_test_sparse_depth.txt
│        │   └── kitti_test_validity_map.txt
│        ├── training
│        │   ├── kitti_train_ground_truth-clean.txt
│        │   ├── kitti_train_ground_truth.txt
│        │   ├── kitti_train_image-clean.txt
│        │   ├── kitti_train_image.txt
│        │   ├── kitti_train_intrinsics-clean.txt
│        │   ├── kitti_train_intrinsics.txt
│        │   ├── kitti_train_sparse_depth-clean.txt
│        │   ├── kitti_train_sparse_depth.txt
│        │   ├── kitti_train_validity_map-clean.txt
│        │   ├── kitti_train_validity_map.txt
│        │   ├── kitti_unused_ground_truth.txt
│        │   ├── kitti_unused_image.txt
│        │   ├── kitti_unused_intrinsics_depth.txt
│        │   ├── kitti_unused_sparse_depth.txt
│        │   └── kitti_unused_validity_map.txt
│        └── validation
│            ├── kitti_val_ground_truth.txt
│            ├── kitti_val_image.txt
│            ├── kitti_val_intrinsics.txt
│            ├── kitti_val_sparse_depth.txt
│            └── kitti_val_validity_map.txt
│   └── train_val_test_file_paths_raw.zip
├── kitti-calibration
│   └── calib_cam_to_cam.txt
└── nlspn

nlspn
├── depth_selection -> /path/to/kitti-depth-completion/data_depth_selection
├── train
│   ├── 2011_09_26_drive_0001_sync
│       ├── calib_cam_to_cam.txt
│       ├── calib_imu_to_velo.txt
│       ├── calib_velo_to_cam.txt
│       ├── image_02 -> /path/to/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02
│       ├── image_03 -> /path/to/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_03
│       ├── oxts -> /path/to/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/oxts
│       └── proj_depth
└── val
    ├── 2011_09_26_drive_0002_sync
       ├── calib_cam_to_cam.txt
       ├── calib_imu_to_velo.txt
       ├── calib_velo_to_cam.txt
       ├── image_02 -> /path/to/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0002_sync/image_02
       ├── image_03 -> /path/to/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0002_sync/image_03
       ├── oxts -> /path/to/code_that_needs_cv_data/nlspn/data/kitti_raw_data/2011_09_26/2011_09_26_drive_0002_sync/oxts
       └── proj_depth
```

depth_completion_symlinks.ipynb can be used to create symlinks to duplicated parts of the dataset from above
nlspn_prepare_KITTI_DC.py creates symlinks to match the NLSPN format.

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

- calibrated-backprojection-network - submodule used in both conditional_diffusion and rsl_depth_completion with some necessary tooling for depth completion
- ip_basic - submodule used in rsl_depth_completion with utils to interpolate sparse depth maps

To install them, run:

```bash
git submodule update --init --recursive
cd submodules || exit 1
for d in */ ; do (cd "$d" && pip install --no-deps -e .); done
```

## Usage

For the conditional diffusion model for depth completion, see the README in the conditional_diffusion directory: [conditional_diffusion/README.md](conditional_diffusion/README.md).

For the self-supervised depth completion pipeline, see the README in the rsl_depth_completion directory: [rsl_depth_completion/README.md](rsl_depth_completion/README.md).

## Benchmarking

See the README in the benchmarking directory: [benchmarking/README.md](benchmarking/README.md).

## Acknowledgements

- <https://github.com/fangchangma/self-supervised-depth-completion> - base code for data loading and self-supervised losses.
- <https://github.com/alexklwong/calibrated-backprojection-network> - base code for KBnet KITTI dataset.
- <https://github.com/ashleve/lightning-hydra-template#best-practices> - template for the project structure.
