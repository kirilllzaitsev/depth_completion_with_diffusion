# Official repository for the paper TBA

## **NOTE!** The master branch is outdated and is no longer updated. The **main** branch is 'main'.

## Contents

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

## Usage

poetry run -m self_supervised_dc.main

## TODO

- [ ] Add code for working with NYUv2 dataset
- [ ] Abstract out the data loading code from the concrete benchmark. Ultimate goal - to load several subsequent frames from a video, apply transformations, provide as an input to the network.
- [ ] Create training module
- [ ] Create error-analysis module. This module should answer to quantitative and qualitative questions about the model performance. As many questions as possible.
- [ ] Create inference module
- [ ] Create logging module

## Acknowledgements

- https://github.com/fangchangma/self-supervised-depth-completion - base code for data loading and self-supervised losses.
- https://github.com/alexklwong/calibrated-backprojection-network - base code for additional loss functions.