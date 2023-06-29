# Conditional Diffusion Model for Depth Completion

## Setup

- create .env file with the following content:

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

Notebooks:

```bash
conditional_diffusion
└── warm-start-with-ssl.ipynb <- experiments with self-supervised fine-tuning
├── sampling-from-trained-imagen.ipynb <- sampling experiments with Imagen
├── test-trained-models.ipynb <- sampling experiments with Imagen
├── full_forward_path_kbnet.ipynb <- experiments with KBnet
├── sdm_exp.ipynb <- experiments with sparse depth interpolation
├── fitting_stable_diffusion.ipynb <- experiments with stable diffusion
├── image_encoder.ipynb <- experiments with image encoder
├── kitti_sample.ipynb <- sample from KITTI dataset
├── advanced_dms.ipynb <- experiments with diffusion models from huggingface
```

## Training

You can train / fine-tune Imagen; train unconditional / conditional stable diffusion.
See script/*.sh for examples of training commands.

Input arguments can be any of the attributes of the cfg class from config.py (see setup_train_pipeline in pipeline_utils.py for details). Default values are specified in config.py itself. Under configs/ there are YAML files with configs for two training scenarios:

- training Imagen on the full KITTI dataset (full_dataset.yaml)
- overfitting Imagen on a single batch of the KITTI dataset (overfit.yaml)

## Fine-tuning with self-supervision

Main fine-tuning experiments are in the warm-start-with-ssl.ipynb notebook. The notebook is self-contained and can be run on a local machine.

Alternatively, see scripts/fine_tune_imagen_ssl.sh for an example of a fine-tuning using scripts (see train_imagen.py, train_imagen_loop.py).

## Collecting results

### Comet ML

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

### Training artifacts

The last training checkpoint of a model, parameters of the model (see model.py), samples from the model, pickled gradients of the model are physically stored in the logs/logdir_name folder in the following format. For example, a Comet experiment named 'tender_patio_2971' has its corresponding folder in logs/euler/final_results:

```bash
/cluster/home/kzaitse/rsl_depth_completion/rsl_depth_completion/logs/euler/final_results/tender_patio_2971
├── depth_grads.pkl
├── model_params.json
├── sample-2-unet-0.png
├── sample-X-unet-0.png
└── unet-2-last.pt
```
