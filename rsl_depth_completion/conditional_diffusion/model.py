from imagen_pytorch import Imagen, Unet
from rsl_depth_completion.conditional_diffusion.utils import log_params_to_exp


def init_model(timesteps, experiment, ds_kwargs):
    unet_base_params = dict(
        dim=64,
        dim_mults=[1, 1, 2, 2, 4, 4],
        channels=1,
        channels_out=None,
        text_embed_dim=512,
        num_resnet_blocks=2,
        layer_attns=[False, False, False, False, False, True],
        layer_cross_attns=[False, False, False, False, False, True],
        attn_heads=4,
        lowres_cond=True,
        memory_efficient=False,
        attend_at_middle=False,
        cond_dim=None,
        cond_images_channels=cond_image_channels(ds_kwargs),
    )

    unet_base = Unet(**unet_base_params)

    unet_super_res = Unet(
        dim=64,
        dim_mults=[1, 1, 2, 2, 4, 4],
        channels=1,
        channels_out=None,
        text_embed_dim=512,
        num_resnet_blocks=2,
        layer_attns=[False, False, False, False, False, True],
        layer_cross_attns=[False, False, False, False, False, True],
        attn_heads=4,
        lowres_cond=True,
        memory_efficient=False,
        attend_at_middle=False,
        cond_dim=None,
        cond_images_channels=cond_image_channels(ds_kwargs)
    )

    unets = [unet_base, unet_super_res]

    imagen_params = dict(
        text_embed_dim=512,
        channels=1,
        timesteps=timesteps,
        loss_type="l2",
        lowres_sample_noise_level=0.2,
        dynamic_thresholding_percentile=0.9,
        only_train_unet_number=None,
        image_sizes=[64, 128],
        text_encoder_name="google/t5-v1_1-base",
        auto_normalize_img=False,
        cond_drop_prob=0.1,
        condition_on_text=ds_kwargs["use_text_embed"],
        pred_objectives="noise",
    )
    imagen = Imagen(unets=unets, **imagen_params)

    experiment.log_parameters({f"imagen_{k}": v for k, v in imagen_params.items()})
    experiment.log_parameters(
        {f"unet_base_{k}": v for k, v in unet_base_params.items()}
    )
    log_params_to_exp(experiment, unet_base_params, "unet_base_params")
    log_params_to_exp(experiment, imagen_params, "imagen_params")

    return unets, imagen


def cond_image_channels(ds_kwargs):
    if not ds_kwargs["use_cond_image"]:
        return 0
    return 3 if ds_kwargs["use_rgb_as_cond_image"] else 1
