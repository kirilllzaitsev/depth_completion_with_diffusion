import torch
from rsl_depth_completion.diffusion.configs import DiffusionConfig
from rsl_depth_completion.diffusion.utils import extract


# forward diffusion
def q_sample(x_start, t, diffusion_config: DiffusionConfig, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(
        diffusion_config.sqrt_alphas_cumprod, t, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_config.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
