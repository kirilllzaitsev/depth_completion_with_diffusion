import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from rsl_depth_completion.conditional_diffusion.custom_trainer import ImagenTrainer
from torchvision.utils import save_image
from tqdm import tqdm


def train(cfg, trainer: ImagenTrainer, train_dataloader, train_writer, out_dir):
    progress_bar = tqdm(total=cfg.num_epochs, disable=False)
    batch_size = train_dataloader.batch_size

    eval_batch = next(iter(train_dataloader))
    with train_writer.as_default():
        log_batch(eval_batch, epoch=1, batch_size=batch_size, prefix="eval")

    global_step = 0

    for epoch in range(cfg.num_epochs):
        progress_bar.set_description(f"Epoch {epoch}")
        running_loss = {"loss": 0, "diff_to_orig_img": 0}
        if cfg.do_overfit:
            data_gen = enumerate(train_dataloader)
        else:
            data_gen = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in data_gen:
            images = batch["image"].to(cfg.device)
            if "text_embed" in batch:
                text_embeds = batch["text_embed"].to(cfg.device)
            else:
                text_embeds = None
            if "cond_image" in batch:
                cond_images = batch["cond_image"].to(cfg.device)
            else:
                cond_images = None
            for i in range(1, len(trainer.imagen.unets) + 1):
                loss = trainer(
                    images=images,
                    text_embeds=text_embeds,
                    cond_images=cond_images,
                    unet_number=i,
                )
                trainer.update(unet_number=i)

                running_loss["loss"] += loss

            with train_writer.as_default():
                tf.summary.scalar(
                    "step/loss",
                    loss,
                    step=global_step,
                )
                if cfg.do_overfit and cfg.do_save_inputs_every_batch:
                    log_batch(batch, epoch, len(images), prefix="train")
            if cfg.do_overfit and batch_idx == 0:
                break
            global_step += 1

        with train_writer.as_default():
            tf.summary.scalar("epoch/loss", running_loss["loss"], step=epoch)

        progress_bar.update(1)

        if (epoch - 1) % cfg.sampling_freq == 0 or epoch == cfg.num_epochs - 1:
            progress_bar.set_postfix(**running_loss)

            if cfg.do_sample:
                eval_text_embeds = (
                    eval_batch["text_embed"].to(cfg.device)
                    if "text_embed" in eval_batch
                    else None
                )
                eval_cond_images = (
                    eval_batch["cond_image"].to(cfg.device)
                    if "cond_image" in eval_batch
                    else None
                )

                samples = trainer.sample(
                    text_embeds=eval_text_embeds,
                    cond_images=eval_cond_images,
                    cond_scale=cfg.cond_scale,
                    batch_size=batch_size,
                    stop_at_unet_number=None,
                    return_all_unet_outputs=True,
                )

                with train_writer.as_default():
                    # relies on return_all_unet_outputs=True
                    for unet_idx in range(len(samples)):
                        out_path = f"{out_dir}/sample-{epoch}-unet-{unet_idx}.png"
                        save_image(samples[unet_idx], str(out_path), nrow=10)
                        name = f"samples/unet_{unet_idx}" if unet_idx > 0 else "samples"
                        tf.summary.image(
                            name,
                            samples[unet_idx]
                            .cpu()
                            .detach()
                            .numpy()
                            .transpose(0, 2, 3, 1),
                            max_outputs=batch_size,
                            step=global_step,
                        )
            if cfg.train_one_epoch:
                break


def log_batch(batch, epoch, batch_size, prefix=None):
    train_img_name = "train_img"
    sdm_name = "sdm"
    rgb_name = "rgb"
    if prefix is not None:
        train_img_name = f"{prefix}/{train_img_name}"
        sdm_name = f"{prefix}/{sdm_name}"
        rgb_name = f"{prefix}/{rgb_name}"

    tf.summary.image(
        train_img_name,
        batch["image"].cpu().numpy().transpose(0, 2, 3, 1),
        max_outputs=batch_size,
        step=epoch,
    )
    tf.summary.image(
        sdm_name,
        batch["sdm"].cpu().numpy().transpose(0, 2, 3, 1),
        max_outputs=batch_size,
        step=epoch,
    )
    tf.summary.image(
        rgb_name,
        batch["rgb"].cpu().numpy().transpose(0, 2, 3, 1),
        max_outputs=batch_size,
        step=epoch,
    )
