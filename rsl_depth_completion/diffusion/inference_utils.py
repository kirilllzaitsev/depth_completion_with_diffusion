from dataclasses import dataclass

import torch
from rsl_depth_completion.diffusion.configs import DiffusionConfig
from rsl_depth_completion.diffusion.diffusion_utils import extract
from tqdm import tqdm




@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_config: DiffusionConfig):
    betas_t = extract(diffusion_config.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_config.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(diffusion_config.sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(diffusion_config.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, shape, timesteps, diffusion_config: DiffusionConfig):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps
    ):
        img = p_sample(
            model,
            img,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
            diffusion_config,
        )
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(
    model,
    img_size,
    timesteps,
    diffusion_config: DiffusionConfig,
    channels,
    batch_size=16,
):
    return p_sample_loop(
        model,
        shape=(batch_size, channels, *img_size),
        timesteps=timesteps,
        diffusion_config=diffusion_config,
    )
