# Self-supervised depth completion pipeline

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

Control over the training process is provided by the [Hydra](https://hydra.cc/) framework. Comprehensive instructions on how to use the pipeline efficiently are at [this link](https://github.com/ashleve/lightning-hydra-template). The configuration files are located in the configs/ directory.

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

## Add a new model

To add a new model for benchmarking purposes, please follow the process described for the KBnet.

1. Create a LightningModule that mirrors runtime of the model in the official code (models/benchmarking/adapters/kbnet_module.py)
2. Create a Hydra config file that provides necessary arguments for the LightningModule (configs/models/kbnet.yaml)
3. Create a datamodule that provides the data for the model (datamodules/kitti.py)
4. Create a Hydra config file that provides necessary arguments for the data module (configs/data/kbnet.yaml)

**Note.** KBnet uses non-standard KITTI dataset: following official implementation, RGB images must be stored as triplets at times (t-1, t, t+1), not as individual frames in original KITTI dataset. For this reason, the datamodule for KBnet is different from the datamodule for other models that use KITTI dataset (data/kitti_datamodule.py). If a model uses standard KITTI dataset, please reuse the kitti_datamodule.py.

## Test individual LightningModule or LightningDataModule

An example of testing a module separately from the pipeline is in rsl_depth_completion/data/kitti_datamodule.py. The mechanism is the same for LightningModule, and includes creating a main function, wrapping it with a hydra.main decorator pointing to an appropriate config file.

## Run a pipeline

To train/val/test a model, run:

```bash
poetry run python rsl_depth_completion/train.py
```

To get predictions for the test set, run:

```bash
poetry run python rsl_depth_completion/eval.py
```

## Training and evaluation workflows

Each execution of **train.py** append data to the log file of the same name. Depending on the experiment setup (chosen callbacks, extras, etc.) in the train.yaml

As a result, a new directory with the execution timestamp in name will be created under the logs/train/runs. If the train script succeeds, the directory will have a similar structure to the following:

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

- checkpoints - directory with model checkpoints at the end of each epoch and the last checkpoint
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