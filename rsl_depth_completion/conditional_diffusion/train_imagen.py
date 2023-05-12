import gc
import itertools
import os
import shutil
from pathlib import Path

import comet_ml
import tensorflow as tf
import torch
import torch.optim as optim
from load_data import load_data
from model import init_model
from rsl_depth_completion.conditional_diffusion.config import cfg as cfg_cls
from rsl_depth_completion.conditional_diffusion.custom_trainer import ImagenTrainer
from rsl_depth_completion.conditional_diffusion.train import train
from rsl_depth_completion.conditional_diffusion.utils import (
    dict2mdtable,
    log_params_to_exp,
)
from rsl_depth_completion.diffusion.utils import set_seed

cfg = cfg_cls(path=cfg_cls.default_file)

set_seed(cfg.seed)
torch.backends.cudnn.benchmark = True

if cfg.is_cluster:
    if not os.path.exists(f"{cfg.tmpdir}/cluster"):
        os.system(f"tar -xvf /cluster/project/rsl/kzaitsev/dataset.tar -C {cfg.tmpdir}")


gc.collect()
torch.cuda.empty_cache()


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


ds_params = product_dict(
    use_text_embed=[True, False],
    use_cond_image=[True, False],
    use_rgb_as_cond_image=[True, False],
)

logdir = Path("./logs") if not cfg.is_cluster else Path(cfg.tmpdir) / "logs"
if cfg.do_overfit:
    logdir = logdir / "standalone_trainer"
else:
    logdir = logdir / "train"

# shutil.rmtree(logdir, ignore_errors=True)


best_params = {
    "kitti": {
        "use_text_embed": False,
        "use_cond_image": True,
        "use_rgb_as_cond_image": False,
    },
    "mnist": {
        "use_text_embed": True,
        "use_cond_image": False,
        "use_rgb_as_cond_image": True,
    },
}

ds_kwargs = best_params[cfg.ds_name]

ds_kwargs["use_rgb_as_text_embed"] = not ds_kwargs["use_rgb_as_cond_image"]
ds_kwargs["include_sdm_and_rgb_in_sample"] = True
ds_kwargs["do_crop"] = True
print(ds_kwargs)

ds, train_dataloader, val_dataloader = load_data(
    ds_name=cfg.ds_name, do_overfit=cfg.do_overfit, **ds_kwargs
)

experiment = comet_ml.Experiment(
    api_key="W5npcWDiWeNPoB2OYkQvwQD0C",
    project_name="rsl_depth_completion",
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_tensorboard_logging=True,
    log_env_details=True,
    log_env_host=False,
    log_env_gpu=True,
    log_env_cpu=True,
    disabled=cfg.disabled,
)
log_params_to_exp(experiment, ds_kwargs, "dataset")
log_params_to_exp(experiment, cfg.params(), "base_config")

print(
    "Number of train samples",
    len(train_dataloader) * train_dataloader.batch_size,
)


unets, model = init_model(experiment, ds_kwargs, cfg)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    "Number of parameters in model",
    num_params,
)

input_name = "interp_sdm"

if ds_kwargs["use_cond_image"]:
    if ds_kwargs["use_rgb_as_cond_image"]:
        img_cond = "rgb"
    else:
        img_cond = "sdm"
else:
    img_cond = "none"

if ds_kwargs["use_text_embed"]:
    if ds_kwargs["use_rgb_as_text_embed"]:
        text_cond = "rgb"
    else:
        text_cond = "sdm"
else:
    text_cond = "none"

cond = f"{img_cond=}_{text_cond=}"
exp_dir = f"{input_name=}/{cond=}/{cfg.lr=}_{cfg.timesteps=}"

exp_dir = f"{len(os.listdir(logdir)) + 1:03d}" if os.path.isdir(logdir) else "001"
train_logdir = logdir / exp_dir / cond
train_logdir.mkdir(parents=True, exist_ok=True)
train_writer = tf.summary.create_file_writer(str(train_logdir))

trainer = ImagenTrainer(model, use_lion=False, lr=cfg.lr)


train(
    cfg,
    trainer,
    train_dataloader,
    out_dir=train_logdir,
    train_writer=train_writer,
)

with train_writer.as_default():
    tf.summary.text(
        "hyperparams",
        dict2mdtable({**ds_kwargs, **cfg.params(), "num_params": num_params}),
        1,
    )

experiment.add_tags([k for k, v in ds_kwargs.items() if v])
if hasattr(cfg, "other_tags"):
    experiment.add_tags(cfg.other_tags)
experiment.add_tag("imagen")
experiment.add_tag(cfg.ds_name)
experiment.add_tag("overfit" if cfg.do_overfit else "full_data")
# experiment.add_tag("debug" if cfg.do_debug else "train")

experiment.end()
