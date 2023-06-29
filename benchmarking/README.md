# Benchmarking

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

## Tool to analyze depth estimation results

The tool is based on the Mask2Former model. Before using it, take a look at the scripts/extract_masks.sh and download the model specified for MODEL.WEIGHTS. Demo file kitti_val_image.txt contains the list of images to be processed. **The image paths in the file must exist.**

### Setup

- install calibrated-backprojection-network package from the rsl_depth_completion/submodules
- update paths in common.yaml

### Usage

1. Run the script to extract masks from the images:

```bash
bash scripts/extract_masks.sh
```

2. Open benchmarking_eval.ipynb and run the cells.

The results will be saved in the output directory specified in the notebook (based on Tensorboard). Models' mean absolute error (in meters) per-category will be saved to the file model_mae_per_category.csv.
