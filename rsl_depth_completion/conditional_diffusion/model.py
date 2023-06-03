from rsl_depth_completion.conditional_diffusion.config import cfg
from rsl_depth_completion.conditional_diffusion.custom_imagen_pytorch import NullUnet
from rsl_depth_completion.conditional_diffusion.utils import log_params_to_exp


def init_model(experiment, ds_kwargs, cfg: cfg):
    # unet_base_params = dict(
    #     dim=cfg.dim,
    #     dim_mults=[1, 1, 2, 2, 4, 4],
    #     channels=1,
    #     channels_out=None,
    #     text_embed_dim=512,
    #     num_resnet_blocks=cfg.num_resnet_blocks,
    #     layer_attns=[False, False, True, True, True, True],
    #     layer_cross_attns=[False, False, True, True, True, True],
    #     attn_heads=6,
    #     lowres_cond=False,
    #     memory_efficient=cfg.memory_efficient,
    #     attend_at_middle=False,
    #     cond_dim=None,
    #     cond_images_channels=cond_image_channels(ds_kwargs),
    # )
    unet_base_params = dict(
        dim=cfg.base_dim,
        dim_mults=(1, 2, 3, 4),
        num_resnet_blocks=cfg.num_resnet_blocks,
        # num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True),
        attn_heads=8,
        ff_mult=2.0,
        memory_efficient=False,
        channels=1,
        cond_images_channels=cond_image_channels(ds_kwargs),
    )

    if cfg.use_triplet_loss:
        from rsl_depth_completion.conditional_diffusion.custom_imagen_pytorch_ssl import (
            Imagen,
            Unet,
        )
    else:
        from rsl_depth_completion.conditional_diffusion.custom_imagen_pytorch import (
            Imagen,
            Unet,
        )

    if cfg.only_super_res:
        unet_base = NullUnet()
        print("Using NullUnet as base unet")
    else:
        unet_base = Unet(**unet_base_params)

    unets = [unet_base]
    input_imgsize = cfg.input_img_size
    assert input_imgsize[0] == input_imgsize[1], "Only square images supported"
    image_sizes = cfg.unets_output_res

    if cfg.only_base:
        image_sizes = [image_sizes[0]]
        
    if cfg.use_super_res:
        unet_super_res = Unet(
            dim=cfg.sr_dim,
            dim_mults=[1, 1, 2, 2, 4, 4],
            channels=1,
            channels_out=None,
            text_embed_dim=512,
            num_resnet_blocks=cfg.num_resnet_blocks,
            layer_attns=[False, False, True, True, True, True],
            layer_cross_attns=[False, False, True, True, True, True],
            attn_heads=6,
            lowres_cond=False,
            memory_efficient=cfg.memory_efficient,
            attend_at_middle=False,
            cond_dim=None,
            cond_images_channels=cond_image_channels(ds_kwargs),
        )

        unets.append(unet_super_res)

    imagen_params = dict(
        text_embed_dim=512,
        channels=1,
        timesteps=cfg.timesteps,
        loss_type="l2",
        lowres_sample_noise_level=0.2,
        dynamic_thresholding_percentile=0.9,
        only_train_unet_number=None,
        image_sizes=image_sizes,
        text_encoder_name="google/t5-v1_1-base",
        auto_normalize_img=cfg.auto_normalize_img,
        cond_drop_prob=0.1,
        condition_on_text=ds_kwargs["use_text_embed"],
        pred_objectives="noise",
    )
    imagen = Imagen(cfg=cfg, unets=unets, **imagen_params)

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
