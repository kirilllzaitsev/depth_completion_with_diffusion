import os
import shutil

from load_data import load_data
from model import init_model
from rsl_depth_completion.conditional_diffusion.pipeline_utils import (
    create_tracking_exp,
    get_ds_kwargs,
    setup_train_pipeline,
)
from rsl_depth_completion.conditional_diffusion.train_imagen_loop import train_loop
from rsl_depth_completion.conditional_diffusion.utils import log_params_to_exp


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--logdir-name", default="comet")
    args, _ = parser.parse_known_args()
    use_ssl = args.use_ssl

    cfg, logdir = setup_train_pipeline(args.logdir_name, use_ssl=use_ssl)
    experiment = create_tracking_exp(cfg)

    if cfg.disabled:
        exp_dir = (
            f"{len(os.listdir(logdir)) + 1:03d}" if os.path.isdir(logdir) else "001"
        )
        exp_dir += f"_{cfg.exp_targets=}"
    else:
        exp_dir = experiment.name
    train_logdir = logdir / exp_dir
    train_logdir.mkdir(parents=True, exist_ok=True)

    ds_kwargs = get_ds_kwargs(cfg)

    ds, train_dataloader, val_dataloader = load_data(
        ds_name=cfg.ds_name, do_overfit=cfg.do_overfit, cfg=cfg, **ds_kwargs
    )

    experiment.add_tag("imagen")
    src_files = [
        os.path.basename(__file__),
        "train_imagen_loop.py",
        "custom_imagen_pytorch.py",
        "custom_trainer.py",
    ]
    for src_file in src_files:
        experiment.log_code(src_file)

    num_samples = len(train_dataloader) * train_dataloader.batch_size
    unets, model = init_model(experiment, ds_kwargs, cfg, str(train_logdir))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_params_to_exp(experiment, {**ds_kwargs, "num_samples": num_samples}, "dataset")
    log_params_to_exp(
        experiment,
        {**cfg.params(), "num_params": num_params, "train_logdir": train_logdir},
        "base_config",
    )

    print(
        "Number of train samples",
        num_samples,
    )

    print(
        "Number of parameters in model",
        num_params,
    )

    trainer_kwargs = dict(
        imagen=model,
        use_lion=False,
        lr=cfg.lr,
        max_grad_norm=1.0,
        fp16=cfg.fp16,
        use_ema=False,
        accelerate_log_with="comet_ml",
        accelerate_project_dir="logs",
    )

    train_loop_kwargs = dict(
        cfg=cfg,
        train_dataloader=train_dataloader,
        out_dir=train_logdir,
        experiment=experiment,
        trainer_kwargs=trainer_kwargs,
        eval_batch=ds.eval_batch,
        use_ssl=use_ssl,
    )
    train_loop(**train_loop_kwargs)

    experiment.add_tag("completed")
    experiment.end()

    if "debug" in cfg.other_tags:
        shutil.rmtree(train_logdir, ignore_errors=True)
        print(f"Deleted {train_logdir}")


if __name__ == "__main__":
    #  = args.

    main()
