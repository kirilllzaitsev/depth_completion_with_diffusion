import os
import typing as t

import comet_ml
import torch
from rsl_depth_completion.conditional_diffusion.custom_trainer import ImagenTrainer
from rsl_depth_completion.conditional_diffusion.custom_trainer_ssl import (
    ImagenTrainer as ImagenTrainerSSL,
)
from rsl_depth_completion.conditional_diffusion.utils import log_batch
from torchvision.utils import save_image
from tqdm import tqdm

ImagenTrainerType = t.Union[ImagenTrainer, ImagenTrainerSSL]


def train_loop(
    cfg,
    train_dataloader,
    experiment: comet_ml.Experiment,
    out_dir,
    trainer_kwargs,
    eval_batch,
    use_ssl=False,
):
    eval_batch_size = eval_batch["input_img"].shape[0]

    log_batch(
        eval_batch,
        step=1,
        batch_size=eval_batch_size,
        experiment=experiment,
        prefix="eval",
        max_depth=cfg.max_depth,
    )

    if cfg.only_super_res:
        start_at_unet_number = 2
    else:
        start_at_unet_number = 1

    num_unets = len(trainer_kwargs["imagen"].unets)
    if cfg.only_base:
        stop_at_unet_number = 1
    else:
        stop_at_unet_number = num_unets + 1

    for unet_idx in range(start_at_unet_number, stop_at_unet_number):
        trainer = init_trainer(trainer_kwargs, use_ssl)
        train_loop_single_unet(
            cfg,
            trainer,
            train_dataloader,
            experiment,
            out_dir,
            eval_batch,
            unet_idx,
            save_path=f"{out_dir}/unet-{unet_idx}-last.pt",
        )
    if not (cfg.only_base or cfg.only_super_res):
        global_step = (stop_at_unet_number) * cfg.num_epochs * len(train_dataloader) + 1
        if cfg.do_sample:
            sample(
                cfg,
                trainer,
                experiment,
                out_dir,
                eval_batch,
                global_step,
                start_at_unet_number=start_at_unet_number,
                stop_at_unet_number=num_unets + 1,
            )


def init_trainer(trainer_kwargs, use_ssl):
    if use_ssl:
        trainer = ImagenTrainerSSL(**trainer_kwargs)
    else:
        trainer = ImagenTrainer(**trainer_kwargs)
    return trainer


def train_loop_single_unet(
    cfg,
    trainer: ImagenTrainerType,
    train_dataloader,
    experiment,
    out_dir,
    eval_batch,
    unet_idx,
    save_path,
):
    global_step = 0
    progress_bar = tqdm(total=cfg.num_epochs, disable=False)
    batch_size = train_dataloader.batch_size

    for epoch in range(cfg.num_epochs):
        progress_bar.set_description(f"Unet {unet_idx}\tEpoch {epoch}")
        running_loss = {"loss": 0, "diff_to_orig_img": 0}
        if cfg.do_overfit:
            batch_scale_factor = batch_size // eval_batch["input_img"].shape[0]
            batch_scale_factor = max(1, batch_scale_factor)
            data_gen = enumerate(
                [
                    {
                        k: torch.repeat_interleave(v, batch_scale_factor, dim=0)
                        for k, v in eval_batch.items()
                    }
                ]
                * len(train_dataloader)
            )
        else:
            data_gen = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in data_gen:
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
            forwards_kwargs = dict(
                images=images,
                text_embeds=text_embeds,
                cond_images=cond_images,
                unet_number=unet_idx,
                max_batch_size=cfg.max_batch_size,
                validity_map_depth=validity_map_depth
                if unet_idx == (trainer.num_unets) and cfg.use_validity_map_depth
                else None,
            )
            if cfg.use_triplet_loss:
                adj_imgs = batch["adj_imgs"]
                ssl_kwargs = dict(
                    image0=batch["rgb"],
                    image1=adj_imgs[0],
                    image2=adj_imgs[1],
                    filtered_sparse_depth0=batch["sdm"],
                    filtered_validity_map_depth0=validity_map_depth,
                    intrinsics=batch["intrinsics"],
                )
                forwards_kwargs.update(ssl_kwargs)

            if cfg.use_triplet_loss:
                loss, output_depth = trainer(**forwards_kwargs)
            else:
                loss = trainer(**forwards_kwargs)

            trainer.update(unet_number=unet_idx)

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
                if not cfg.do_save_last_model:
                    print(
                        f"Ignoring {save_path} and saving epoch-wise to {out_dir}/model-{epoch}.pt"
                    )
                    save_path = save_path.replace(".pt", f"-{epoch}.pt")
                trainer.save(save_path)
                print(f"Saved trainer of unet {unet_idx} to {save_path}")

            if cfg.do_sample:
                if cfg.only_super_res:
                    start_image_or_video = eval_batch["lowres_img"]
                    start_at_unet_number = unet_idx
                else:
                    start_image_or_video = None
                    start_at_unet_number = 1
                sample(
                    cfg,
                    trainer,
                    experiment,
                    out_dir,
                    eval_batch,
                    global_step,
                    start_at_unet_number=start_at_unet_number,
                    start_image_or_video=start_image_or_video,
                    stop_at_unet_number=unet_idx,
                )
            if cfg.train_one_epoch:
                break


def sample(
    cfg,
    trainer: ImagenTrainerType,
    experiment,
    out_dir,
    batch,
    global_step,
    stop_at_unet_number,
    start_at_unet_number=1,
    start_image_or_video=None,
):
    """
    start_at_unet_number: must be used in context of available samples from a base unet
    """
    eval_text_embeds = batch["text_embed"] if "text_embed" in batch else None
    eval_cond_images = batch["cond_img"] if "cond_img" in batch else None
    batch_size = batch["input_img"].shape[0]

    samples = trainer.sample(
        text_embeds=eval_text_embeds,
        cond_images=eval_cond_images,
        cond_scale=cfg.cond_scale,
        batch_size=batch_size,
        start_at_unet_number=start_at_unet_number,
        stop_at_unet_number=stop_at_unet_number,
        start_image_or_video=start_image_or_video,
        return_all_unet_outputs=True,
    )

    if len(samples[0]) > 1:
        experiment.log_metric(
            "epoch/abs_diff_btw_samples",
            torch.sum(torch.abs(samples[0][0] - samples[0][1])).item(),
            step=global_step,
        )

    for unet_idx_samples in range(len(samples)):
        out_path = f"{out_dir}/sample-{global_step}-unet-{unet_idx_samples}.png"
        save_image(samples[unet_idx_samples], str(out_path), nrow=10)
        name = f"samples/unet_{unet_idx_samples+start_at_unet_number-1}"
        unet_samples = (
            samples[unet_idx_samples].cpu().detach().numpy().transpose(0, 2, 3, 1)
        )
        for idx in range(len(samples[unet_idx_samples])):
            experiment.log_image(
                unet_samples[idx],
                f"{name}_{idx}",
                step=global_step,
            )
    return samples
