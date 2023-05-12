import gc
import os
from dataclasses import dataclass
from pathlib import Path

import comet_ml
import tensorflow as tf
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from rsl_depth_completion.conditional_diffusion.config import cfg as cfg_cls
from rsl_depth_completion.conditional_diffusion.load_data import load_data
from rsl_depth_completion.conditional_diffusion.train import log_batch
from rsl_depth_completion.conditional_diffusion.utils import (
    dict2mdtable,
    log_params_to_exp,
)
from rsl_depth_completion.diffusion.utils import set_seed
from torchvision.utils import save_image
from tqdm.auto import tqdm

cfg = cfg_cls(path=cfg_cls.default_file)
cfg.timesteps = 300


set_seed(cfg.seed)
torch.backends.cudnn.benchmark = True

if cfg.is_cluster:
    if not os.path.exists(f"{cfg.tmpdir}/cluster"):
        os.system(f"tar -xvf /cluster/project/rsl/kzaitsev/dataset.tar -C {cfg.tmpdir}")


gc.collect()
torch.cuda.empty_cache()

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


@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = train_dataloader.batch_size
    eval_batch_size = (
        train_dataloader.batch_size
    )  # how many images to sample during evaluation
    num_epochs = cfg.num_epochs
    gradient_accumulation_steps = 1
    learning_rate = cfg.lr
    lr_warmup_steps = 500
    save_image_epochs = 2
    save_model_epochs = 1000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "stable-diffusion"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

    num_train_timesteps = cfg.timesteps
    num_inference_timesteps = cfg.timesteps


config = TrainingConfig()


model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(
        128,
        256,
        512,
        512,
    ),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
    ),
)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    "Number of parameters in model",
    num_params,
)

sample_image = ds[0]["image"].unsqueeze(0)
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

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
# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=config.lr_warmup_steps,
#     num_training_steps=(len(train_dataloader) * config.num_epochs),
# )
lr_scheduler = None


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline, output_type="pil"):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=config.num_inference_timesteps,
        output_type=output_type,
    ).images

    # Make a grid out of the images
    # image_grid = make_grid(images, rows=4, cols=4)

    # # Save the images
    # test_dir = os.path.join(config.output_dir, "samples")
    # os.makedirs(test_dir, exist_ok=True)
    # image_grid.save(f"{test_dir}/{epoch:04d}.png")
    return images


def train_loop(
    train_writer,
    config: TrainingConfig,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    out_dir,
):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs"),
    )
    # if accelerator.is_main_process:
    #     if config.output_dir is not None:
    #         os.makedirs(config.output_dir, exist_ok=True)
    #     accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    # model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, lr_scheduler
    # )

    global_step = 0

    progress_bar_epoch = tqdm(total=cfg.num_epochs, disable=False)

    eval_batch = next(iter(train_dataloader))
    batch_size = config.train_batch_size
    with train_writer.as_default():
        log_batch(eval_batch, epoch=1, batch_size=batch_size, prefix="eval")

    # Now you train the model
    for epoch in range(cfg.num_epochs):
        progress_bar_epoch.set_description(f"Epoch {epoch}")
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=True or not accelerator.is_local_main_process,
        )
        running_loss = {"loss": 0, "diff_to_orig_img": 0}
        if cfg.do_overfit:
            data_gen = enumerate(train_dataloader)
        else:
            data_gen = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        running_loss = {"loss": 0, "diff_to_orig_img": 0}

        for batch_idx, batch in data_gen:
            clean_images = batch["image"]
            if "text_embed" in batch:
                text_embeds = batch["text_embed"].to(clean_images.device)
            else:
                text_embeds = None
            if "cond_image" in batch:
                cond_images = batch["cond_image"].to(clean_images.device)
            else:
                cond_images = None
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # with accelerator.accumulate(model):
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            # accelerator.backward(loss)

            # accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            loss = loss.detach().item()
            running_loss["loss"] += loss

            with train_writer.as_default():
                tf.summary.scalar(
                    "batch/loss",
                    loss,
                    step=global_step,
                )
                if cfg.do_overfit and cfg.do_save_inputs_every_batch:
                    log_batch(batch, epoch, batch_size, prefix="train")
            if cfg.do_overfit and batch_idx == 0:
                break

            progress_bar.update(1)
            logs = {
                "loss": loss,
                "lr": optimizer.optimizer.param_groups[0]["lr"],
                "step": global_step,
            }
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )

            with train_writer.as_default():
                tf.summary.scalar("epoch/loss", running_loss["loss"], step=epoch)

            progress_bar_epoch.update(1)

            if (epoch - 1) % cfg.sampling_freq == 0 or epoch == cfg.num_epochs - 1:
                progress_bar_epoch.set_postfix(**running_loss)

                if cfg.do_sample:
                    eval_text_embeds = (
                        eval_batch["text_embed"].to(clean_images.device)
                        if "text_embed" in eval_batch
                        else None
                    )
                    eval_cond_images = (
                        eval_batch["cond_image"].to(clean_images.device)
                        if "cond_image" in eval_batch
                        else None
                    )

                    # samples = trainer.sample(
                    #     text_embeds=eval_text_embeds,
                    #     cond_images=eval_cond_images,
                    #     cond_scale=cfg.cond_scale,
                    #     batch_size=batch_size,
                    #     stop_at_unet_number=None,
                    #     return_all_unet_outputs=True,
                    # )
                    samples = evaluate(config, epoch, pipeline, output_type="numpy")

                    with train_writer.as_default():
                        # relies on return_all_unet_outputs=True
                        # out_path = f"{out_dir}/sample-{epoch}.png"
                        # save_image(torch.from_numpy(samples), str(out_path), nrow=10)
                        name = "samples"
                        tf.summary.image(
                            name,
                            samples,
                            max_outputs=batch_size,
                            step=epoch,
                        )
                if cfg.train_one_epoch:
                    break

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


# notebook_launcher(train_loop, args, num_processes=1)


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

logdir = Path("./logs") if not cfg.is_cluster else Path(cfg.tmpdir) / "logs"
if cfg.do_overfit:
    logdir = logdir / config.output_dir
else:
    logdir = logdir / "train"
exp_dir = f"{len(os.listdir(logdir)) + 1:03d}" if os.path.isdir(logdir) else "001"
train_logdir = logdir / exp_dir / cond
train_logdir.mkdir(parents=True, exist_ok=True)
train_writer = tf.summary.create_file_writer(str(train_logdir))

args = (
    train_writer,
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    train_logdir,
)
train_loop(*args)

with train_writer.as_default():
    tf.summary.text(
        "hyperparams",
        dict2mdtable({**ds_kwargs, **cfg.params(), "num_params": num_params}),
        1,
    )

experiment.add_tags([k for k, v in ds_kwargs.items() if v])
if hasattr(cfg, "other_tags"):
    experiment.add_tags(cfg.other_tags)
experiment.add_tag(cfg.ds_name)
experiment.add_tag("stable-diffusion")
experiment.add_tag("overfit" if cfg.do_overfit else "full_data")
# experiment.add_tag("debug" if cfg.do_debug else "train")

experiment.end()
