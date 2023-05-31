import os

import comet_ml
from load_data import load_data
from model import init_model
from rsl_depth_completion.conditional_diffusion.pipeline_utils import (
    create_tracking_exp,
    get_ds_kwargs,
    setup_train_pipeline,
)
from rsl_depth_completion.conditional_diffusion.train_imagen_loop import train_loop
from rsl_depth_completion.conditional_diffusion.train_imagen_ssl_loop import (
    train_loop as train_loop_ssl,
)
from rsl_depth_completion.conditional_diffusion.utils import log_params_to_exp


def main():
    logdir_name = "debug"
    # logdir_name = "standalone_trainer"
    cfg, train_logdir = setup_train_pipeline(logdir_name)
    # cfg.disabled = True

    ds_kwargs = get_ds_kwargs(cfg)

    ds, train_dataloader, val_dataloader = load_data(
        ds_name=cfg.ds_name, do_overfit=cfg.do_overfit, cfg=cfg, **ds_kwargs
    )

    experiment = create_tracking_exp(cfg)
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
    unets, model = init_model(experiment, ds_kwargs, cfg)
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
    if cfg.use_triplet_loss:
        from rsl_depth_completion.conditional_diffusion.custom_trainer_ssl import ImagenTrainer
    else:
        from rsl_depth_completion.conditional_diffusion.custom_trainer import ImagenTrainer

    trainer = ImagenTrainer(**trainer_kwargs)

    train_loop_kwargs = dict(
        cfg=cfg,
        trainer=trainer,
        train_dataloader=train_dataloader,
        out_dir=train_logdir,
        experiment=experiment,
        trainer_kwargs=trainer_kwargs,
        eval_batch=ds.eval_batch,
    )
    if cfg.use_triplet_loss:
        train_loop_ssl(**train_loop_kwargs)
    else:
        train_loop(**train_loop_kwargs)

    experiment.add_tag("completed")
    experiment.end()


if __name__ == "__main__":
    #  = args.

    main()
