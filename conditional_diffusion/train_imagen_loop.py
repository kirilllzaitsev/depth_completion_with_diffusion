import copy
import os
import pickle
import typing as t

import comet_ml
import torch
from rsl_depth_completion.conditional_diffusion.custom_trainer import ImagenTrainer
from rsl_depth_completion.conditional_diffusion.custom_trainer_ssl import (
    ImagenTrainer as ImagenTrainerSSL,
)
from rsl_depth_completion.conditional_diffusion.kbnet_utils import load_kbnet
from rsl_depth_completion.conditional_diffusion.ssl_utils import (
    calc_error_to_gt,
    plot_train_depths_overall,
)
from rsl_depth_completion.conditional_diffusion.utils import (
    log_batch,
    plot_full_prediction,
    print_metrics,
)
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
        stop_at_unet_number = 2
    else:
        stop_at_unet_number = num_unets + 1

    for unet_idx in range(start_at_unet_number, stop_at_unet_number):
        trainer = init_trainer(trainer_kwargs, use_ssl)
        if cfg.trainer_ckpt_path is not None:
            trainer.load(cfg.trainer_ckpt_path, only_model=True)
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
                out_dir,
                eval_batch,
                global_step,
                start_at_unet_number=start_at_unet_number,
                stop_at_unet_number=num_unets + 1,
                experiment=experiment,
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
    progress_bar = tqdm(total=cfg.num_epochs, disable=False, leave=False)
    batch_size = train_dataloader.batch_size

    parameters_pose_model = list(trainer.pose_model.parameters())
    parameters_depth_model = list(trainer.parameters())
    depth_grads = {k: [] for k in range(len(parameters_depth_model))}
    pose_grads = {k: [] for k in range(len(parameters_pose_model))}
    rec_losses = {
        "image_01_loss": [],
        "image_02_loss": [],
    }
    losses = {
        "loss_color": [],
        "loss_structure": [],
        "loss_sparse_depth": [],
        "loss_smoothness": [],
        "triplet_loss": [],
        "imagen_loss": [],
        "MAE": [],
        "RMSE": [],
        "IMAE": [],
        "IRMSE": [],
    }
    val_losses = {
        "MAE": [],
        "RMSE": [],
        "IMAE": [],
        "IRMSE": [],
    }
    num_train_samples_to_watch = 4
    output_depths = [[] for _ in range(num_train_samples_to_watch)]
    sampled_depths = []
    kbnet_predictor = load_kbnet()

    for epoch in range(cfg.num_epochs):
        progress_bar.set_description(f"Unet {unet_idx}\tEpoch {epoch}")
        running_loss = {"loss": 0}
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
            data_gen = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                leave=False,
                desc="Train",
            )
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
                    image1=adj_imgs[:, 0],
                    image2=adj_imgs[:, 1],
                    filtered_sparse_depth0=batch["sdm"],
                    filtered_validity_map_depth0=validity_map_depth,
                    intrinsics=batch["intrinsics"],
                    rec_losses=rec_losses,
                    losses=losses,
                )
                forwards_kwargs.update(ssl_kwargs)

            if cfg.use_triplet_loss:
                loss, output_depth = trainer(**forwards_kwargs)
                for i in range(len(parameters_depth_model)):
                    if parameters_depth_model[i].grad is not None:
                        depth_grads[i].append(
                            torch.sum(torch.abs(parameters_depth_model[i].grad)).item()
                        )
                for i in range(len(parameters_pose_model)):
                    if parameters_pose_model[i].grad is not None:
                        pose_grads[i].append(
                            torch.sum(torch.abs(parameters_pose_model[i].grad)).item()
                        )

                trainer.update(unet_number=unet_idx)

                mae, rmse, imae, irmse = calc_error_to_gt(output_depth, batch["gt"])

                losses["MAE"].append(mae.item())
                losses["RMSE"].append(rmse.item())
                losses["IMAE"].append(imae.item())
                losses["IRMSE"].append(irmse.item())

                for stats in [losses, rec_losses]:
                    for k, v in stats.items():
                        experiment.log_metric(f"step/{k}", v[-1], step=global_step)

                if batch_idx in range(num_train_samples_to_watch):
                    # take first sample from first N batches
                    output_depths[batch_idx].append(output_depth[0])

                    full_pred_figs = plot_full_prediction(
                        eval_batch=batch,
                        output_depths=output_depth,
                        kbnet_predictor=kbnet_predictor,
                        # idx_to_use=-1,
                    )
                    for idx, fig in enumerate(full_pred_figs):
                        experiment.log_figure(
                            f"step/val/sample_{idx}",
                            fig,
                            step=global_step,
                        )
            else:
                loss = trainer(**forwards_kwargs)

                experiment.log_metric(
                    "step/loss",
                    loss,
                    step=global_step,
                )

            for i in range(len(parameters_depth_model)):
                if parameters_depth_model[i].grad is not None:
                    depth_grads[i].append(
                        torch.sum(torch.abs(parameters_depth_model[i].grad)).item()
                    )

            trainer.update(unet_number=unet_idx)

            running_loss["loss"] += loss

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
                upd_save_path = save_path
                if cfg.do_save_every_n_epochs:
                    upd_save_path = upd_save_path.replace(".pt", f"-{epoch}.pt")
                trainer.save(upd_save_path)
                print(f"Saved trainer of unet {unet_idx} to {upd_save_path}")

            if cfg.do_sample:
                if cfg.only_super_res:
                    start_image_or_video = eval_batch["lowres_img"]
                    start_at_unet_number = unet_idx
                else:
                    start_image_or_video = None
                    start_at_unet_number = 1
                sampled_depths = sample(
                    cfg,
                    trainer,
                    out_dir,
                    eval_batch,
                    global_step,
                    start_at_unet_number=start_at_unet_number,
                    start_image_or_video=start_image_or_video,
                    stop_at_unet_number=unet_idx,
                    experiment=experiment,
                )
                assert len(sampled_depths) == 1, "Only the last unet should sample"
                sampled_depth = sampled_depths[0]
                sampled_depths.append(sampled_depth)
                if cfg.use_triplet_loss:
                    mae, rmse, imae, irmse = calc_error_to_gt(
                        sampled_depth, eval_batch["gt"]
                    )

                    val_losses["MAE"].append(mae.item())
                    val_losses["RMSE"].append(rmse.item())
                    val_losses["IMAE"].append(imae.item())
                    val_losses["IRMSE"].append(irmse.item())

                    for k, v in val_losses.items():
                        experiment.log_metric(f"step/val/{k}", v[-1], step=global_step)

                    full_pred_figs = plot_full_prediction(
                        eval_batch=eval_batch,
                        output_depths=sampled_depth,
                        kbnet_predictor=kbnet_predictor,
                        # idx_to_use=-1,
                    )
                    for idx, fig in enumerate(full_pred_figs):
                        experiment.log_figure(
                            f"step/val/sample_{idx}",
                            fig,
                            step=global_step,
                        )

            with open(f"{out_dir}/depth_grads.pkl", "wb") as f:
                pickle.dump(depth_grads, f)
            if cfg.train_one_epoch:
                break
    if cfg.do_sample:
        training_progression_figs = plot_train_depths_overall(
            output_depths, out_dir, "train"
        )
        for idx, fig in enumerate(training_progression_figs):
            experiment.log_figure(
                f"step/val/sample_{idx}",
                fig,
                step=global_step + 1,
            )
        sampling_progression_figs = plot_train_depths_overall(
            sampled_depths, out_dir, "eval"
        )
        for idx, fig in enumerate(sampling_progression_figs):
            experiment.log_figure(
                f"step/val/sample_{idx}",
                fig,
                step=global_step + 1,
            )
    print_metrics(
        losses["MAE"],
        losses["RMSE"],
        losses["IMAE"],
        losses["IRMSE"],
        comment="SSL train",
    )
    print_metrics(
        val_losses["MAE"],
        val_losses["RMSE"],
        val_losses["IMAE"],
        val_losses["IRMSE"],
        comment="SSL val",
    )
    return global_step


def sample(
    cfg,
    trainer: ImagenTrainerType,
    out_dir,
    batch,
    global_step,
    stop_at_unet_number,
    start_at_unet_number=1,
    start_image_or_video=None,
    experiment=None,
):
    """
    start_at_unet_number: must be used when base unet is trained and provides low resolution conditioning, i.e., start_image_or_video
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
    for i, unet_samples in enumerate(samples):
        samples[i] = unet_samples.detach().cpu()

    if len(samples[0]) > 1 and experiment is not None:
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
            samples[unet_idx_samples].numpy().transpose(0, 2, 3, 1)
        )
        if experiment is not None:
            for idx in range(len(samples[unet_idx_samples])):
                experiment.log_image(
                    unet_samples[idx],
                    f"{name}_{idx}",
                    step=global_step,
                )
    # samples are in the range [0, 1]
    for unet_idx_samples in range(len(samples)):
        samples[unet_idx_samples] *= trainer.max_predict_depth
    return samples
