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
- Replace '???' with PATH_TO_DEPTH_COMPLETION in all files under the self_supervised_dc/configs/dataset directory.

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

### Conditional diffusion

### Self-supervised depth completion

At this point, the following parts of the pipeline are implemented:

- KITTI dataset with the following sample components:
  - RGB frame
  - Sparse depth map
  - camera intrinsics (calibration matrix)
  - camera extrinsics (rotation, translation)
  - adjacent RGB frames
  - ground truth depth map
- train/val/test and predict for the KBnet model

Control over the training process is provided by the [Hydra](https://hydra.cc/) framework. The configuration files are located in the configs/ directory.

#### Add a new model

To add a new model for benchmarking purposes, please follow the process described for the KBnet.

1. Create a LightningModule that mirrors runtime of the model in the official code (models/benchmarking/adapters/kbnet_module.py)
2. Create a Hydra config file that provides necessary arguments for the LightningModule (configs/models/kbnet.yaml)
3. Create a datamodule that provides the data for the model (datamodules/kitti.py)
4. Create a Hydra config file that provides necessary arguments for the data module (configs/data/kbnet.yaml)

**Note.** KBnet uses non-standard KITTI dataset: following official implementation, RGB images must be stored as triplets at times (t-1, t, t+1), not as individual frames in original KITTI dataset. For this reason, the datamodule for KBnet is different from the datamodule for other models that use KITTI dataset (data/kitti_datamodule.py). If a model uses standard KITTI dataset, please reuse the kitti_datamodule.py.

#### Run a pipeline

To train/val/test a model, run:

```bash
poetry run -m rsl_depth_completion.train
```

## Acknowledgements

- https://github.com/fangchangma/self-supervised-depth-completion - base code for data loading and self-supervised losses.
- https://github.com/alexklwong/calibrated-backprojection-network - base code for additional loss functions.
- https://github.com/ashleve/lightning-hydra-template#best-practices - template for the project structure.