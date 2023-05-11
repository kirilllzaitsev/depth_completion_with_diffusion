import gc
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from rsl_depth_completion.conditional_diffusion.config import cfg as cfg_cls
from rsl_depth_completion.conditional_diffusion.load_data import load_data
from rsl_depth_completion.diffusion.utils import set_seed
from tqdm.auto import tqdm

cfg = cfg_cls(path="configs/overfit.yaml")

set_seed(cfg.seed)
torch.backends.cudnn.benchmark = True

if cfg.is_cluster:
    if not os.path.exists(f"{cfg.tmpdir}/cluster"):
        os.system(f"tar -xvf /cluster/project/rsl/kzaitsev/dataset.tar -C {cfg.tmpdir}")


gc.collect()
torch.cuda.empty_cache()


ds_name = "mnist"

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

ds_kwargs = best_params[ds_name]

ds_kwargs["use_rgb_as_text_embed"] = not ds_kwargs["use_rgb_as_cond_image"]
ds_kwargs["include_sdm_and_rgb_in_sample"] = True
ds_kwargs["do_crop"] = True
print(ds_kwargs)

ds, train_dataloader, val_dataloader = load_data(
    ds_name=ds_name, do_overfit=cfg.do_overfit, **ds_kwargs
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
    save_model_epochs = 10
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
        128,
        256,
        256,
        512,
        512,
    ),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")


sample_image = ds[0]["image"].unsqueeze(0)
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

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


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=config.num_inference_timesteps,
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(
    config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler
):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=True or not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if step == 1:
                break

            clean_images = batch["image"]
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

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": optimizer.optimizer.param_groups[0]["lr"],
                "step": global_step,
            }
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )

            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                print(f"Epoch: {epoch}\t{logs}")
                evaluate(config, epoch, pipeline)

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

# notebook_launcher(train_loop, args, num_processes=1)
config.num_epochs = 2000
config.save_image_epochs = 400
config.eval_batch_size = 2
train_loop(*args)
