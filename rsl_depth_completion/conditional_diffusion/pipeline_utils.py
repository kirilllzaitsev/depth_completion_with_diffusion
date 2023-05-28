import argparse
import os
from pathlib import Path

import comet_ml
import torch
from rsl_depth_completion.conditional_diffusion.config import cfg as cfg_cls


def create_tracking_exp(cfg):
    experiment = comet_ml.Experiment(
        api_key="W5npcWDiWeNPoB2OYkQvwQD0C",
        project_name="rsl_depth_completion",
        auto_output_logging="simple",
        auto_metric_logging=True,
        auto_param_logging=True,
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

    experiment.add_tags(["cluster" if cfg.is_cluster else "local"])
    experiment.add_tags(
        [cfg.ds_name, "overfit" if cfg.do_overfit else "full_data"]
    )
    if cfg.other_tags:
        experiment.add_tags(cfg.exp_targets)
    if cfg.num_epochs == 1:
        experiment.add_tag("debug")
    return experiment


def setup_optimizations():
    torch.backends.cudnn.benchmark = True


def setup_train_pipeline():
    from rsl_depth_completion.diffusion.utils import set_seed

    cfg = cfg_cls(path=cfg_cls.default_file)

    parser = argparse.ArgumentParser()
    for attr_key, attr_value in cfg_cls.__dict__.items():
        attr_type = type(attr_value)
        if not attr_key.startswith("__") and not callable(attr_value):
            obj_attr_value = (
                getattr(cfg, attr_key) if attr_key in vars(cfg) else attr_value
            )
            default = obj_attr_value
            arg_name = f"--{attr_key}"
            if attr_type == bool:
                parser.add_argument(arg_name, action="store_true", default=default)
            elif attr_type == list:
                parser.add_argument(arg_name, nargs="*", default=default)
            else:
                parser.add_argument(arg_name, type=attr_type, default=default)
    args, _ = parser.parse_known_args()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    set_seed(cfg.seed)

    if cfg.is_cluster:
        if not os.path.exists(f"{cfg.tmpdir}/cluster"):
            os.system(
                f"tar -xvf /cluster/project/rsl/kzaitsev/dataset.tar -C {cfg.tmpdir} > /dev/null 2>&1"
            )

    logdir = Path("./logs") if not cfg.is_cluster else Path(cfg.cluster_logdir)
    logdir = logdir / "standalone_trainer"

    # shutil.rmtree(logdir, ignore_errors=True)
    exp_dir = f"{len(os.listdir(logdir)) + 1:03d}" if os.path.isdir(logdir) else "001"
    exp_dir += f"_{cfg.exp_targets=}"
    train_logdir = logdir / exp_dir
    train_logdir.mkdir(parents=True, exist_ok=True)
    return cfg, train_logdir


def get_ds_kwargs(cfg):
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
    return ds_kwargs
