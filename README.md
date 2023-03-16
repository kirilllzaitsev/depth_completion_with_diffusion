# Official repository for the paper TBA

## Contents

## Installation

### Pre-requisites

- Python >=3.7
- Poetry ~=1.4.0

```bash
python --version
curl -sSL https://install.python-poetry.org | python -
```

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