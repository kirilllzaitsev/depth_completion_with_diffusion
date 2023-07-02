from dataclasses import dataclass

import torch


@dataclass
class DiffusionConfig:
    betas: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor
