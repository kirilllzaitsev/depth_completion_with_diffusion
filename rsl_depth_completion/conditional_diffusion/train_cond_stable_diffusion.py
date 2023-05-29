import gc
import os
from dataclasses import dataclass
from pathlib import Path

import comet_ml
import tensorflow as tf
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMPipeline,
    DDPMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import HfFolder, Repository, whoami
from load_data import load_data
from model import init_model
from PIL import Image
from rsl_depth_completion.conditional_diffusion.config import cfg as cfg_cls
from rsl_depth_completion.conditional_diffusion.custom_trainer import ImagenTrainer
from rsl_depth_completion.conditional_diffusion.load_data import load_data
from rsl_depth_completion.conditional_diffusion.pipeline_utils import (
    create_tracking_exp,
    get_ds_kwargs,
    setup_train_pipeline,
)
from rsl_depth_completion.conditional_diffusion.train import log_batch, train
from rsl_depth_completion.conditional_diffusion.utils import (
    dict2mdtable,
    log_batch,
    log_params_to_exp,
    rescale_img_to_zero_one_range,
)
from rsl_depth_completion.diffusion.utils import set_seed
from torchvision.utils import save_image
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    train_batch_size = None
    eval_batch_size = None  # how many images to sample during evaluation
    num_epochs = None
    learning_rate = None
    gradient_accumulation_steps = 1
    lr_warmup_steps = 500
    save_model_epochs = 200
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = None

    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

    num_train_timesteps = None
    num_inference_timesteps = None
    model_alias = "cond-stable-diffusion"


def main():
    cfg, train_logdir = setup_train_pipeline(logdir_name=TrainingConfig.model_alias)

    ds_kwargs = get_ds_kwargs(cfg)

    ds, train_dataloader, val_dataloader = load_data(
        ds_name=cfg.ds_name, do_overfit=cfg.do_overfit, cfg=cfg, **ds_kwargs
    )

    experiment = create_tracking_exp(cfg)
    experiment.add_tag("cond-stable-diffusion")
    experiment.log_code(os.path.basename(__file__))

    config = TrainingConfig()
    config.num_train_timesteps = cfg.timesteps
    config.num_inference_timesteps = cfg.timesteps
    config.train_batch_size = cfg.batch_size
    config.eval_batch_size = cfg.batch_size
    config.num_epochs = cfg.num_epochs
    config.learning_rate = cfg.lr
    config.output_dir = train_logdir

    model = UNet2DConditionModel(
        sample_size=64,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        cross_attention_dim=256,
        encoder_hid_dim=512,
        attention_head_dim=6,
        block_out_channels=(
            64,
            128,
            256,
        ),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "CrossAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
        ),
    )

    num_samples = len(train_dataloader) * train_dataloader.batch_size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_params_to_exp(experiment, {**ds_kwargs, "num_samples": num_samples}, "dataset")
    log_params_to_exp(
        experiment,
        {
            **cfg.params(),
            **vars(config),
            "num_params": num_params,
            "train_logdir": train_logdir,
        },
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

    sample_image = ds[0]["input_img"].unsqueeze(0)
    sample_text_embeds = ds[0]["text_embed"].unsqueeze(0)
    print("Input shape:", sample_image.shape)
    print(
        "Output shape:",
        model(
            sample_image, timestep=0, encoder_hidden_states=sample_text_embeds
        ).sample.shape,
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    noise = torch.randn(sample_image.shape)
    timesteps = torch.LongTensor([10])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    Image.fromarray(
        ((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5)
        .type(torch.uint8)
        .numpy()[0][:, :, -1],
        "L",
    )

    print("did the test")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = dict(
        cfg=cfg,
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        experiment=experiment,
        eval_batch=ds.eval_batch,
    )
    train_loop(**args)

    experiment.add_tag("completed")
    experiment.end()


def train_loop(
    cfg: cfg_cls,
    config: TrainingConfig,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    experiment: comet_ml.Experiment,
    eval_batch,
):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="comet_ml",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        # accelerator.init_trackers(
        #     experiment.project_name,
        #     init_kwargs={
        #         "comet_ml": {
        #             "disabled": cfg.disabled,
        #             "api_key": experiment.api_key,
        #             "experiment_key": experiment.get_key(),
        #             "resume": True,
        #         },
        #     },
        # )
        # accelerator.trackers[0].writer = experiment

    batch_size = train_dataloader.batch_size

    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    vae = AutoencoderKL(in_channels=1, out_channels=1, latent_channels=1)
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        unet=model,
        vae=vae,
    )
    pipeline.to(accelerator.device)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    progress_bar_epoch = tqdm(total=cfg.num_epochs, disable=False)

    if accelerator.is_local_main_process:
        log_batch(
            eval_batch,
            step=1,
            batch_size=batch_size,
            experiment=experiment,
            prefix="eval",
            max_depth=cfg.max_depth,
        )
    global_step = 0
    # Now you train the model
    for epoch in range(cfg.num_epochs):
        progress_bar_epoch.set_description(f"Epoch {epoch}")
        progress_bar_batch = tqdm(
            total=len(train_dataloader),
            disable=cfg.do_overfit or not accelerator.is_local_main_process,
        )
        running_loss = {"loss": 0}

        for batch_idx, batch in enumerate(train_dataloader):
            if cfg.do_overfit:
                batch = eval_batch
            images = batch["input_img"].to(accelerator.device)
            if "text_embed" in batch:
                text_embeds = batch["text_embed"].to(accelerator.device)
            else:
                text_embeds = None
            if "cond_img" in batch:
                cond_images = batch["cond_img"]
            else:
                cond_images = None

            validity_map_depth = torch.where(
                batch["sdm"] > 0, torch.ones_like(batch["sdm"]), batch["sdm"]
            ).bool()
            noise = torch.randn(images.shape).to(images.device)
            bs = images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss = loss.detach().item()
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

            progress_bar_batch.update(1)
            logs = {
                "batch_loss": loss,
                "step": global_step,
            }
            progress_bar_batch.set_postfix(**logs)
            if accelerator.is_local_main_process:
                experiment.log_metric("batch/loss", loss, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            experiment.log_metric("epoch/loss", running_loss["loss"], step=epoch)

            progress_bar_epoch.update(1)

            if (epoch - 1) % cfg.sampling_freq == 0 or epoch == cfg.num_epochs - 1:
                progress_bar_epoch.set_postfix(**running_loss)

                if cfg.do_sample:
                    eval_text_embeds = (
                        eval_batch["text_embed"].to(accelerator.device)
                        if "text_embed" in eval_batch
                        else None
                    )
                    eval_cond_images = (
                        eval_batch["cond_img"].to(accelerator.device)
                        if "cond_img" in eval_batch
                        else None
                    )

                    samples = evaluate(
                        config,
                        epoch,
                        pipeline,
                        eval_cond_images,
                        eval_text_embeds,
                        output_type="np",
                    )

                    unet_idx = 0
                    name = f"samples/unet_{unet_idx}"
                    for i, sample in enumerate(samples):
                        experiment.log_image(
                            sample.transpose(2,0,1),
                            f"{name}_{i}",
                            step=global_step,
                        )
                if cfg.train_one_epoch:
                    break

                if (
                    epoch + 1
                ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline.save_pretrained(config.output_dir)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline, image, encoder_hidden_states, output_type="pil"):
    images = pipeline(
        # batch_size=config.eval_batch_size,
        image=image,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=config.num_inference_timesteps,
        output_type=output_type,
        prompt_embeds=encoder_hidden_states,
        negative_prompt_embeds=torch.zeros_like(encoder_hidden_states),
    ).images
    return images


if __name__ == "__main__":
    main()
