import json

import cv2
import numpy as np
import torch
from rsl_depth_completion.conditional_diffusion import utils as data_utils
from rsl_depth_completion.conditional_diffusion.img_utils import (
    center_crop,
    fix_channel_not_last,
    resize,
)
from rsl_depth_completion.conditional_diffusion.utils import load_extractors


class BaseDMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        use_rgb_as_text_embed,
        use_rgb_as_cond_image,
        use_cond_image=True,
        use_text_embed=True,
        include_sdm_and_rgb_in_sample=True,
        cond_img_sdm_interpolation_mode=None,
        input_img_sdm_interpolation_mode=None,
        eval_batch=None,
        max_depth=80,
        target_lowres_img_size=None,
        *args,
        **kwargs,
    ):
        self.cfg = cfg
        self.use_rgb_as_text_embed = use_rgb_as_text_embed
        self.use_rgb_as_cond_image = use_rgb_as_cond_image
        self.use_cond_image = use_cond_image
        self.use_text_embed = use_text_embed
        self.target_lowres_img_size = target_lowres_img_size
        self.extractor_model, self.extractor_processor = load_extractors()
        self.max_depth = max_depth
        self.cond_img_sdm_interpolation_mode = cond_img_sdm_interpolation_mode
        self.input_img_sdm_interpolation_mode = input_img_sdm_interpolation_mode
        self.include_sdm_and_rgb_in_sample = include_sdm_and_rgb_in_sample
        if cond_img_sdm_interpolation_mode == "kbnet":
            self.depth_model = self.load_kbnet_model()
        self.eval_batch = self.prep_eval_batch(eval_batch) if eval_batch else None

    def prep_eval_batch(self, eval_batch):
        if self.cfg.do_crop:
            for k in ["sdm", "rgb", "gt"]:
                eval_batch[k] = self.prep_img(eval_batch[k])
        eval_batch["input_img"] = self.prep_sparse_dm(
            eval_batch["sdm"], self.input_img_sdm_interpolation_mode, channel_dim=1
        )

        if self.use_cond_image:
            if self.use_rgb_as_cond_image:
                rgb = eval_batch["rgb"]
                cond_image = self.prep_rgbs_as_cond_img(rgb)
            else:
                cond_image = self.prep_sdms_as_cond_img(eval_batch["sdm"])
            if len(cond_image.shape) == 3:
                cond_image = cond_image.unsqueeze(0)
            eval_batch["cond_img"] = cond_image
        if self.use_text_embed:
            embeds = []
            if self.use_rgb_as_text_embed:
                for rgb in eval_batch["rgb"]:
                    prepare_pixels = rgb if torch.max(rgb) > 1 else rgb * 255
                    rgb_pixel_values = self.prepare_pixels(prepare_pixels)
                    rgb_embed = self.extractor_model.get_image_features(
                        pixel_values=rgb_pixel_values
                    )
                    embeds.append(rgb_embed)

                # rgb_embeds[-1] = torch.load("random_coco_embed.pt")
                # print("Loaded random coco embed at last index of eval batch")

                eval_batch["text_embed"] = torch.stack(embeds, dim=0).detach()
            else:
                for sdm in eval_batch["sdm"]:
                    prepare_pixels = sdm if torch.max(sdm) > 1 else sdm * self.max_depth
                    sdm_pixel_values = self.prepare_pixels(
                        cv2.cvtColor(
                            prepare_pixels.squeeze().numpy().astype("uint8"),
                            cv2.COLOR_GRAY2RGB,
                        )
                    )
                    sdm_embed = self.extractor_model.get_image_features(
                        pixel_values=sdm_pixel_values
                    )
                    embeds.append(sdm_embed)
            eval_batch["text_embed"] = torch.stack(embeds, dim=0).detach()

        if self.target_lowres_img_size is not None:
            # lowres_batch = torch.load(eval_batch_path)[base_unet_res]
            # eval_batch["lowres_img"] = lowres_batch["input_img"][: self.cfg.batch_size]
            eval_batch["lowres_img"] = self.add_lowres_img(
                eval_batch["sdm"], self.target_lowres_img_size
            )

        return eval_batch

    def prep_rgbs_as_cond_img(self, rgb):
        return rgb if torch.max(rgb) <= 1 else rgb / 255

    def prep_sdms_as_cond_img(self, sdms):
        if self.cond_img_sdm_interpolation_mode == "kbnet":
            sdms = sdms if torch.max(sdms) > 1 else sdms * self.max_depth
            densified_sdms = self.densify_sparse_dm_with_kbnet(sdms)
            cond_image = densified_sdms / densified_sdms.max()
        else:
            sdms = sdms if torch.max(sdms) <= 1 else sdms / self.max_depth
            cond_images = []
            for sdm in sdms:
                cond_images.append(
                    self.prep_sparse_dm(
                        sdm, self.cond_img_sdm_interpolation_mode, channel_dim=0
                    )
                )
            if len(cond_images) == 1:
                return cond_images[0]
            cond_image = torch.stack(cond_images, dim=0)
        return cond_image

    def extend_sample(self, sparse_dm, rgb_image):
        extension = {}
        assert not (
            self.use_rgb_as_cond_image and self.use_rgb_as_text_embed
        ), "Can't use rgb as both cond image and text embed"
        if self.use_cond_image:
            # normalizing because imagen divides by 255 internally if dtype == uint8
            # which is wrong for the case of sparse depth
            if self.use_rgb_as_cond_image:
                cond_image = self.prep_rgbs_as_cond_img(rgb_image)
            else:
                cond_image = self.prep_sdms_as_cond_img(sparse_dm)
            extension["cond_img"] = cond_image.detach()

        if self.use_text_embed:
            # denormalizing because CLIP processor requires pixel values in [0, 255]
            if self.use_rgb_as_text_embed:
                prepare_pixels = (
                    rgb_image if torch.max(rgb_image) > 1 else rgb_image * 255
                )
                prepare_pixels = prepare_pixels.int()
                rgb_pixel_values = self.prepare_pixels(prepare_pixels)
                rgb_embed = self.extractor_model.get_image_features(
                    pixel_values=rgb_pixel_values
                )

                extension["text_embed"] = rgb_embed.detach()
            else:
                prepare_pixels = (
                    sparse_dm
                    if torch.max(sparse_dm) > 1
                    else sparse_dm * self.max_depth
                )
                sdm_pixel_values = self.prepare_pixels(
                    cv2.cvtColor(sparse_dm.squeeze().numpy(), cv2.COLOR_GRAY2RGB)
                )
                sdm_pixel_values = sdm_pixel_values.int()
                sdm_embed = self.extractor_model.get_image_features(
                    pixel_values=sdm_pixel_values
                )
                extension["text_embed"] = sdm_embed.detach()
        if self.include_sdm_and_rgb_in_sample:
            extension["rgb"] = (rgb_image).detach() / 255
            extension["sdm"] = (sparse_dm).detach() * self.max_depth

        if self.target_lowres_img_size is not None:
            extension["lowres_img"] = self.add_lowres_img(
                sparse_dm, self.target_lowres_img_size
            )

        return {**extension}

    def add_lowres_img(self, sparse_dm, target_img_size=None):
        # temporary solution. lowres img should be a dense depth map from the base unet trained with SSL objective
        lowres_img = self.prep_sparse_dm(
            sparse_dm,
            "infill",
            channel_dim=0 if len(sparse_dm.shape) == 3 else 1,
        )
        if target_img_size is not None:
            lowres_img = resize(
                lowres_img,
                target_img_size,
            )
        return lowres_img

    def prepare_pixels(self, img):
        img = fix_channel_not_last(img)
        return self.extractor_processor(
            images=torch.from_numpy(np.array(img)),
            return_tensors="pt",
        ).pixel_values

    def prep_sparse_dm(self, sparse_dms, interpolation_mode, channel_dim=-1):
        if interpolation_mode is None:
            return sparse_dms
        if len(sparse_dms.shape) == 4:
            cond_images = []
            for sdm in sparse_dms:
                cond_images.append(self._prep_sparse_dm(sdm, interpolation_mode))
            cond_images = torch.stack(cond_images, dim=0)
        else:
            cond_images = self._prep_sparse_dm(sparse_dms, interpolation_mode)
        return cond_images.unsqueeze(channel_dim)

    def _prep_sparse_dm(self, sdm, interpolation_mode):
        if interpolation_mode is None:
            return sdm
        return torch.from_numpy(
            data_utils.infill_sparse_depth(sdm.numpy())[0]
            if interpolation_mode == "infill"
            else data_utils.interpolate_sparse_depth(
                sdm.squeeze().numpy(), do_multiscale=True
            )
        )

    def densify_sparse_dm_with_kbnet(self, sdm):
        is_single_sample = len(sdm.shape) == 3
        if is_single_sample:
            sdm = sdm.unsqueeze(0)
        validity_map_depth = torch.zeros_like(sdm)
        validity_map_depth[sdm > 0] = 1
        input_depth = [sdm, validity_map_depth]

        input_depth = torch.cat(input_depth, dim=1)

        input_depth = self.depth_model.sparse_to_dense_pool(input_depth)
        B, _, H, W = sdm.shape
        input_depth = input_depth.permute(0, 2, 3, 1).reshape(B, -1, 8)

        # sdm = sdm.reshape(B, -1, 8)
        u, s, v = torch.pca_lowrank(input_depth, niter=2)
        densified_sdm = (input_depth @ v[..., :1]).reshape(B, -1, H, W).cpu().detach()
        return densified_sdm[0] if is_single_sample else densified_sdm

    def load_kbnet_model(self):
        from argparse import Namespace

        from kbnet.kbnet_model import KBNetModel

        args = Namespace(**json.load(open("kbnet_val_params.json", "r")))

        depth_model = KBNetModel(
            input_channels_image=args.input_channels_image,
            input_channels_depth=args.input_channels_depth,
            min_pool_sizes_sparse_to_dense_pool=args.min_pool_sizes_sparse_to_dense_pool,
            max_pool_sizes_sparse_to_dense_pool=args.max_pool_sizes_sparse_to_dense_pool,
            n_convolution_sparse_to_dense_pool=args.n_convolution_sparse_to_dense_pool,
            n_filter_sparse_to_dense_pool=args.n_filter_sparse_to_dense_pool,
            n_filters_encoder_image=args.n_filters_encoder_image,
            n_filters_encoder_depth=args.n_filters_encoder_depth,
            resolutions_backprojection=args.resolutions_backprojection,
            n_filters_decoder=args.n_filters_decoder,
            deconv_type=args.deconv_type,
            weight_initializer=args.weight_initializer,
            activation_func=args.activation_func,
            min_predict_depth=args.min_predict_depth,
            max_predict_depth=args.max_predict_depth,
            device=args.device,
        )

        # Restore model and set to evaluation mode
        depth_model.restore_model(args.depth_model_restore_path)
        depth_model.eval()
        return depth_model

    def prep_img(self, x, channels_last=False):
        x = center_crop(
            x, crop_size=self.cfg.max_input_img_size, channels_last=channels_last
        )
        return x
