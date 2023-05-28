import os

import comet_ml
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_depth_completion.conditional_diffusion.custom_trainer import ImagenTrainer
from rsl_depth_completion.conditional_diffusion.utils import (
    rescale_img_to_zero_one_range,
)
from torchvision.utils import save_image
from tqdm import tqdm


def train(
    cfg,
    trainer: ImagenTrainer,
    train_dataloader,
    experiment: comet_ml.Experiment,
    out_dir,
    trainer_kwargs,
    eval_batch,
):
    progress_bar = tqdm(total=cfg.num_epochs, disable=False)
    batch_size = train_dataloader.batch_size

    log_batch(
        eval_batch,
        step=1,
        batch_size=batch_size,
        experiment=experiment,
        prefix="eval",
        max_depth=cfg.max_depth,
    )

    global_step = 0
    is_multi_unet_training = (trainer.num_unets) > 1

    for epoch in range(cfg.num_epochs):
        progress_bar.set_description(f"Epoch {epoch}")
        running_loss = {"loss": 0, "diff_to_orig_img": 0}
        if cfg.do_overfit:
            data_gen = enumerate(train_dataloader)
        else:
            data_gen = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in data_gen:
            if cfg.do_overfit:
                batch = eval_batch
            images = batch["input_img"]
            if "text_embed" in batch:
                text_embeds = batch["text_embed"]
            else:
                text_embeds = None
            if "cond_img" in batch:
                cond_images = batch["cond_img"]
            else:
                cond_images = None

            validity_map_depth = torch.where(
                batch["sdm"] > 0, torch.ones_like(batch["sdm"]), batch["sdm"]
            ).bool()
            for i in range(1, (trainer.num_unets) + 1):
                ckpt_path = f"{out_dir}/checkpoint_{i}.pt"
                if is_multi_unet_training:
                    trainer = ImagenTrainer(**trainer_kwargs)
                    if os.path.exists(ckpt_path):
                        trainer.load(ckpt_path)
                loss = trainer(
                    images=images,
                    text_embeds=text_embeds,
                    cond_images=cond_images,
                    unet_number=i,
                    max_batch_size=cfg.max_batch_size,
                    validity_map_depth=validity_map_depth
                    if i == (trainer.num_unets) and cfg.use_validity_map_depth
                    else None,
                )
                trainer.update(unet_number=i)
                if is_multi_unet_training:
                    trainer.save(ckpt_path)

                running_loss["loss"] += loss

            experiment.log_metric(
                "step/loss",
                loss,
                step=global_step,
            )
            if cfg.do_overfit and cfg.do_save_inputs_every_batch:
                log_batch(
                    batch,
                    global_step,
                    batch_size,
                    experiment=experiment,
                    prefix="train",
                )

            global_step += 1

            if cfg.do_overfit and batch_idx == 0:
                break

        experiment.log_metric("epoch/loss", running_loss["loss"], step=global_step)

        progress_bar.update(1)

        if (epoch - 1) % cfg.sampling_freq == 0 or epoch == cfg.num_epochs - 1:
            progress_bar.set_postfix(**running_loss)
            if cfg.do_save_model:
                if cfg.do_save_last_model:
                    save_path = f"{out_dir}/model-last.pt"
                    trainer.save(save_path)
                    print(f"Saved model to {save_path}")
                else:
                    trainer.save(f"{out_dir}/model-{epoch}.pt")

            if cfg.do_sample:
                eval_text_embeds = (
                    eval_batch["text_embed"] if "text_embed" in eval_batch else None
                )
                eval_cond_images = (
                    eval_batch["cond_img"] if "cond_img" in eval_batch else None
                )

                samples = trainer.sample(
                    text_embeds=eval_text_embeds,
                    cond_images=eval_cond_images,
                    cond_scale=cfg.cond_scale,
                    batch_size=batch_size,
                    stop_at_unet_number=cfg.stop_at_unet_number,
                    return_all_unet_outputs=True,
                )

                if len(samples[0]) > 1:
                    experiment.log_metric(
                        "epoch/intersample_abs_diff",
                        torch.sum(torch.abs(samples[0][0] - samples[0][1])).item(),
                        step=global_step,
                    )

                for unet_idx in range(len(samples)):
                    out_path = f"{out_dir}/sample-{epoch}-unet-{unet_idx}.png"
                    save_image(samples[unet_idx], str(out_path), nrow=10)
                    name = f"samples/unet_{unet_idx}"
                    unet_samples = (
                        samples[unet_idx].cpu().detach().numpy().transpose(0, 2, 3, 1)
                    )
                    for idx in range(len(samples[unet_idx])):
                        experiment.log_image(
                            unet_samples[idx],
                            f"{name}_{idx}",
                            step=global_step,
                        )
            if cfg.train_one_epoch:
                break


def log_batch(
    batch,
    step,
    batch_size,
    experiment: comet_ml.Experiment,
    prefix=None,
    max_depth=80.0,
):
    for k, v in batch.items():
        v = v.cpu().numpy().transpose(0, 2, 3, 1)
        v = rescale_img_to_zero_one_range(v)
        for idx in range(batch_size):
            name = f"{k}_{idx}"
            if prefix is not None:
                name = f"{prefix}/{name}"
            experiment.log_image(
                v[idx],
                name,
                step=step,
            )
