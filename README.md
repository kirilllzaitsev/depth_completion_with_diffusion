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

### Conditional diffusion

#### Setup

- create conditional_diffusion/.env file with the following content:

```bash
path_to_project_dir=??? # path to the project directory
base_kitti_dataset_dir=??? # path to the KITTI dataset directory
```

Folder structure:

```bash
conditional_diffusion
├── config.py <- central config for the pipeline
├── configs
│   ├── full_dataset.yaml <- config for trainining using the full KITTI dataset
│   └── overfit.yaml <- config for overfitting a single batch of the KITTI dataset
├── scripts <- scripts to train Imagen, Stable Diffusion
├── custom_imagen_pytorch.py <- Imagen
├── custom_imagen_pytorch_ssl.py <- Imagen with self-supervision part ***
├── custom_trainer.py <- Imagen trainer
├── custom_trainer_ssl.py <- Imagen trainer with self-supervision part ***
├── eval_batch_utils.py <- utils for working with data
├── img_utils.py <- utils for working with images
├── kbnet_utils.py <- utils for working with KBnet
├── load_data_base.py <- base dataset with common logic
├── load_data_kitti.py <- KITTI dataset
├── load_data_mnist.py <- MNIST dataset
├── load_data.py <- main data handling logic
├── model.py <- Imagen definition
├── models
│   └── kbnet <- pretrained KBnet components
├── pipeline_utils.py <- utils for working with the pipeline
├── ssl_utils.py <- utils for working with self-supervision part ***
├── train_cond_stable_diffusion.py
├── train_imagen_loop.py
├── train_imagen.py
├── train_stable_diffusion.py <- training script for stable diffusion conditioned on image embeddings
├── train_uncond_stable_diffusion.py <- training script for unconditional stable diffusion
├── train_utils.py <- utils for model training
└── utils.py <- general utils
```

#### Training

You can train / fine-tune Imagen; train unconditional / conditional stable diffusion.
See script/*.sh for examples of training commands.

Input arguments can be any of the attributes of the cfg class from conditional_diffusion/config.py (see setup_train_pipeline in pipeline_utils.py for details). Default values are specified in config.py itself. Under conditional_diffusion/configs/ there are YAML files with configs for two training scenarios:

- training Imagen on the full KITTI dataset (full_dataset.yaml)
- overfitting Imagen on a single batch of the KITTI dataset (overfit.yaml)

#### Collecting results

##### Comet ML

Metrics, graphics, and some assets are logged to Comet ML.
[Use this link](https://www.comet.com/kirilllzaitsev/rsl-depth-completion?shareable=58IO83O45oWcvPd6pK0CRqdZG) to examine all logged experiments.
The 'Experiments' UI is an entry point to experiments.

![Experiments](assets/experiments.png?raw=true "Experiments")

- 'Columns' allows to display more experiment parameters, e.g., batch size, image resolution, etc.
- 'Filter' is used to filter out experiments by their parameters
- 'Tags' column is a set of keywords that give rough idea of the experiment
- 'Archive' contains experiments that failed for some reason

An individual experiment breaks down into the following (most important) tabs:

- Panels - overview of the important results
![Experiment](assets/experiment.png?raw=true "Experiment")

Note. The Panels from above are configured to use 'ssl' view (see the top right) that displays results relevant to fine-tuning with SSL. However, the pre-training stage requires a different view/layout (e.g., 'pre-train' or 'pre-train-super-res').

- Code - relevant source files
- Hyperparameters - hyperparameters that were logged
- Metrics - metrics that were logged
- System Metrics - CPU / GPU / Memory stats
- Graphics - model inputs, samples, etc.

##### Training artifacts

The last training checkpoint of a model, parameters of the model (see model.py), samples from the model, pickled gradients of the model are physically stored in the conditional_diffusion/logs/logdir_name folder in the following format. For example, a Comet experiment named 'tender_patio_2971' has its corresponding folder in logs/euler/final_results:

```bash
/cluster/home/kzaitse/rsl_depth_completion/rsl_depth_completion/conditional_diffusion/logs/euler/final_results/tender_patio_2971
├── depth_grads.pkl
├── model_params.json
├── sample-2-unet-0.png
├── sample-X-unet-0.png
└── unet-2-last.pt
```

### Self-supervised depth completion pipeline

At this point, the pipeline has the following parts:

- KITTI dataset with the following sample components:
  - RGB frame
  - Sparse depth map
  - camera intrinsics (calibration matrix)
  - camera extrinsics (rotation, translation)
  - adjacent RGB frames
  - ground truth depth map
- KBnet model (models/kbnet_module.py)
- KBnet dataset (data/kbnet_datamodule.py)
- train/val/test and predict for the KBnet model

Control over the training process is provided by the [Hydra](https://hydra.cc/) framework. The configuration files are located in the configs/ directory.

Folder structure:

```bash
rsl_depth_completion
├── data
│   ├── components <- toolkit for data processing
│   ├── __init__.py
│   ├── kbnet
│   │   └── kbnet_dataset.py
│   ├── kbnet_datamodule.py <- KBnet datamodule
│   ├── kitti
│   │   └── kitti_dataset.py <- adapted KITTI dataset
│   └── kitti_datamodule.py <- datamodule for other models that use KITTI dataset
├── eval.py <- evaluation script
├── __init__.py
├── logger.py <- custom Lightning logger
├── metrics.py <- KITTI DC metrics
├── models
│   ├── benchmarking
│   │   ├── calibrated_backprojection_network
│   │   ├── __init__.py
│   │   └── PENet_ICRA2021
│   │       ├── images
│   │       ├── penet
│   │       │   ├── basic.py
│   │       │   ├── CoordConv.py
│   │       │   ├── criteria.py
│   │       │   ├── dataloaders
│   │       │   │   ├── __init__.py
│   │       │   │   ├── kitti_loader.py
│   │       │   │   └── transforms.py
│   │       │   ├── helper.py
│   │       │   ├── __init__.py
│   │       │   ├── main.py
│   │       │   ├── metrics.py
│   │       │   ├── model.py
│   │       │   └── vis_utils.py
│   │       └── setup.py
│   ├── components <- toolkit with module components
│   │   └── __init__.py
│   ├── __init__.py
│   └── kbnet_module.py <- KBnet module
├── trainer.py <- custom Lightning trainer
├── train.py <- main training script
└── utils <- toolkit with utility functions for the pipeline
```

#### Add a new model

To add a new model for benchmarking purposes, please follow the process described for the KBnet.

1. Create a LightningModule that mirrors runtime of the model in the official code (models/benchmarking/adapters/kbnet_module.py)
2. Create a Hydra config file that provides necessary arguments for the LightningModule (configs/models/kbnet.yaml)
3. Create a datamodule that provides the data for the model (datamodules/kitti.py)
4. Create a Hydra config file that provides necessary arguments for the data module (configs/data/kbnet.yaml)

**Note.** KBnet uses non-standard KITTI dataset: following official implementation, RGB images must be stored as triplets at times (t-1, t, t+1), not as individual frames in original KITTI dataset. For this reason, the datamodule for KBnet is different from the datamodule for other models that use KITTI dataset (data/kitti_datamodule.py). If a model uses standard KITTI dataset, please reuse the kitti_datamodule.py.

#### Test individual LightningModule or LightningDataModule

An example of testing a module separately from the pipeline is in rsl_depth_completion/data/kitti_datamodule.py. The mechanism is the same for LightningModule, and includes creating a main function, wrapping it with a hydra.main decorator pointing to an appropriate config file.

#### Run a pipeline

To train/val/test a model, run:

```bash
poetry run python rsl_depth_completion/train.py
```

To get predictions for the test set, run:

```bash
poetry run python rsl_depth_completion/eval.py
```

#### Training and evaluation workflows

Each execution of **train.py** append data to the log file of the same name. Depending on the experiment setup (chosen callbacks, extras, etc.) in the train.yaml

As a result, a new directory with the execution timestamp in name will be created under the logs/train/runs. If the train script succeeds, the directory will have similar structure to the following:

```bash
logs/train/runs/2023-06-28_20-13-46
├── checkpoints
│   ├── epoch_000.ckpt
│   ├── epoch_X.ckpt
│   └── last.ckpt
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── config_tree.log
└── tags.log
```

where:

- checkpoints - directory with model checkpoints
- .hydra - directory with hydra config files
  - config.yaml - config file with all the parameters that aren't interpolated by hydra
  - hydra.yaml - config file with internal hydra parameters
  - overrides.yaml - config file with overrides (if any), i.e., a config property is specified at multiple config files from the hierarchy
- config_tree.log - log file with the config tree containing interpolated parameters
- tags.log - log file with the tags for the experiment

**eval.py** will produce a directory structure under logs/eval/runs that is model-specific. For example, for KBnet, the structure will be as follows:

```bash
logs/eval/runs/2023-06-28_20-25-30
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
└── kbnet_results
    └── test_predictions
        ├── ground_truth
        ├── image
        ├── output_depth
        └── sparse_depth
```

where:

- .hydra - directory with config files (see the one for train.py)
- kbnet_results - directory with evaluation results for KBnet
  - test_predictions - directory with the predictions for the test set
    - ground_truth - directory with the ground truth depth maps (if available)
    - image - directory with the (normalized) RGB images
    - output_depth - directory with the predicted depth maps
    - sparse_depth - directory with the sparse depth maps

## Acknowledgements

- <https://github.com/fangchangma/self-supervised-depth-completion> - base code for data loading and self-supervised losses.
- <https://github.com/alexklwong/calibrated-backprojection-network> - base code for additional loss functions.
- <https://github.com/ashleve/lightning-hydra-template#best-practices> - template for the project structure.
