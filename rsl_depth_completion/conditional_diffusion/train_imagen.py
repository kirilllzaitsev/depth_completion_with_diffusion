import argparse
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

torch.backends.cudnn.benchmark = True


def main():
    cfg = cfg_cls(path=cfg_cls.default_file)

    parser = argparse.ArgumentParser()
    for attr_key, attr_value in cfg_cls.__dict__.items():
        attr_type = type(attr_value)
        if not attr_key.startswith("__") and not callable(attr_value):
            obj_attr_value = getattr(cfg, attr_key) if attr_key in vars(cfg) else attr_value
            if attr_type == bool:
                parser.add_argument(f"--{attr_key}", action="store_true", default=obj_attr_value)
            else:
                parser.add_argument(f"--{attr_key}", type=attr_type, default=obj_attr_value)
    args, _ = parser.parse_known_args()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    set_seed(cfg.seed)

    if cfg.is_cluster:
        if not os.path.exists(f"{cfg.tmpdir}/cluster"):
            os.system(
                f"tar -xvf /cluster/project/rsl/kzaitsev/dataset.tar -C {cfg.tmpdir}"
            )

    logdir = Path("./logs") if not cfg.is_cluster else Path(cfg.cluster_logdir)
    if cfg.do_overfit:
        logdir = logdir / "standalone_trainer"
    else:
        logdir = logdir / "train"

    # shutil.rmtree(logdir, ignore_errors=True)

    best_params = {
        "kitti": {
            "use_text_embed": True,
            "use_cond_image": True,
            "use_rgb_as_cond_image": False,
        },
        "mnist": {
            "use_text_embed": True,
            "use_cond_image": True,
            "use_rgb_as_cond_image": False,
        },
    }

    ds_kwargs = best_params[cfg.ds_name]

    ds_kwargs["use_rgb_as_text_embed"] = not ds_kwargs["use_rgb_as_cond_image"]
    ds_kwargs["include_sdm_and_rgb_in_sample"] = True
    ds_kwargs["do_crop"] = True
    print(ds_kwargs)

    ds, train_dataloader, val_dataloader = load_data(
        ds_name=cfg.ds_name, do_overfit=cfg.do_overfit, cfg=cfg, **ds_kwargs
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

    for code_file in [
        "model.py",
        "load_data.py",
        "load_data_kitti.py",
        "train.py",
        "config.py",
        "custom_imagen_pytorch.py",
    ]:
        experiment.log_asset(code_file, copy_to_tmp=False)

    num_samples = len(train_dataloader) * train_dataloader.batch_size
    unets, model = init_model(experiment, ds_kwargs, cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_params_to_exp(experiment, {**ds_kwargs, "num_samples": num_samples}, "dataset")
    log_params_to_exp(
        experiment, {**cfg.params(), "num_params": num_params}, "base_config"
    )

    if hasattr(cfg, "other_tags"):
        experiment.add_tags(cfg.exp_targets)
    experiment.add_tags(["imagen", cfg.ds_name])
    experiment.add_tag("overfit" if cfg.do_overfit else "full_data")
    if cfg.num_epochs == 1:
        experiment.add_tag("debug")

    print(
        "Number of train samples",
        num_samples,
    )

    print(
        "Number of parameters in model",
        num_params,
    )

    exp_dir = f"{len(os.listdir(logdir)) + 1:03d}" if os.path.isdir(logdir) else "001"
    exp_dir += f"_{cfg.exp_targets=}"
    train_logdir = logdir / exp_dir
    train_logdir.mkdir(parents=True, exist_ok=True)
    train_writer = tf.summary.create_file_writer(str(train_logdir))

    with train_writer.as_default():
        tf.summary.text(
            "hyperparams",
            dict2mdtable({**ds_kwargs, **cfg.params(), "num_params": num_params}),
            1,
        )

    trainer_kwargs = dict(
        imagen=model,
        use_lion=False,
        lr=cfg.lr,
        max_grad_norm=1.0,
        fp16=cfg.fp16,
        use_ema=False,
        accelerate_log_with="tensorboard",
        accelerate_project_dir="logs",
    )
    trainer = ImagenTrainer(**trainer_kwargs)
    trainer.accelerator.init_trackers("train_example")

    try:
        train(
            cfg,
            trainer,
            train_dataloader,
            out_dir=train_logdir,
            train_writer=train_writer,
            trainer_kwargs=trainer_kwargs,
            eval_batch=ds.eval_batch,
        )
    except Exception as e:
        shutil.rmtree(train_logdir)
        raise e

    experiment.end()


if __name__ == "__main__":
    #  = args.

    main()
