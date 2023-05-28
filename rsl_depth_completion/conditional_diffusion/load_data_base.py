import cv2
import numpy as np
import torch
from rsl_depth_completion.conditional_diffusion import utils as data_utils
from rsl_depth_completion.conditional_diffusion.utils import load_extractors


class BaseDMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        use_rgb_as_text_embed,
        use_rgb_as_cond_image,
        use_cond_image=True,
        use_text_embed=True,
        do_crop=True,
        include_sdm_and_rgb_in_sample=True,
        cond_img_sdm_interpolation_mode=None,
        input_img_sdm_interpolation_mode=None,
        eval_batch=None,
        max_depth=80,
        *args,
        **kwargs,
    ):
        self.use_rgb_as_text_embed = use_rgb_as_text_embed
        self.use_rgb_as_cond_image = use_rgb_as_cond_image
        self.use_cond_image = use_cond_image
        self.use_text_embed = use_text_embed
        self.extractor_model, self.extractor_processor = load_extractors()
        self.max_depth = max_depth
        self.do_crop = do_crop
        self.cond_img_sdm_interpolation_mode = cond_img_sdm_interpolation_mode
        self.input_img_sdm_interpolation_mode = input_img_sdm_interpolation_mode
        self.include_sdm_and_rgb_in_sample = include_sdm_and_rgb_in_sample
        self.eval_batch = self.prep_eval_batch(eval_batch) if eval_batch else None
        # self.eval_batch = None

    def prep_eval_batch(self, eval_batch):
        eval_batch["input_img"] = self.prep_sparse_dm(
            eval_batch["sdm"], self.input_img_sdm_interpolation_mode
        )

        if self.use_cond_image:
            if self.use_rgb_as_cond_image:
                cond_image = (
                    eval_batch["rgb"]
                    if torch.max(eval_batch["rgb"]) <= 1
                    else eval_batch["rgb"] / 255
                )
            else:
                sdms = (
                    eval_batch["sdm"]
                    if torch.max(eval_batch["sdm"]) <= 1
                    else eval_batch["sdm"] / self.max_depth
                )
                cond_images = []
                for sdm in sdms:
                    cond_images.append(
                        self.prep_sparse_dm(sdm, self.cond_img_sdm_interpolation_mode)
                    )
                cond_image = torch.stack(cond_images, dim=0)
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
                        cv2.cvtColor(sdm.squeeze().numpy(), cv2.COLOR_GRAY2RGB)
                    )
                    sdm_embed = self.extractor_model.get_image_features(
                        pixel_values=sdm_pixel_values
                    )
                    embeds.append(sdm_embed)
            eval_batch["text_embed"] = torch.stack(embeds, dim=0).detach()
        return eval_batch

    def extend_sample(self, sparse_dm, rgb_image):
        extension = {}
        assert not (
            self.use_rgb_as_cond_image and self.use_rgb_as_text_embed
        ), "Can't use rgb as both cond image and text embed"
        if self.use_cond_image:
            # normalizing because imagen divides by 255 internally if dtype == uint8
            # which is wrong for the case of sparse depth
            if self.use_rgb_as_cond_image:
                cond_image = rgb_image if torch.max(rgb_image) <= 1 else rgb_image / 255
            else:
                cond_image = (
                    sparse_dm
                    if torch.max(sparse_dm) <= 1
                    else sparse_dm / self.max_depth
                )
            extension["cond_img"] = cond_image.detach()

        if self.use_text_embed:
            # denormalizing because CLIP processor requires pixel values in [0, 255]
            if self.use_rgb_as_text_embed:
                prepare_pixels = (
                    rgb_image if torch.max(rgb_image) > 1 else rgb_image * 255
                )
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
                sdm_embed = self.extractor_model.get_image_features(
                    pixel_values=sdm_pixel_values
                )
                extension["text_embed"] = sdm_embed.detach()
        if self.include_sdm_and_rgb_in_sample:
            extension["sdm"] = (sparse_dm).detach()
            extension["rgb"] = (rgb_image).detach()
        return {**extension}

    def prepare_pixels(self, cond_image):
        cond_image = self.fix_channel_not_last(cond_image)
        return self.extractor_processor(
            images=torch.from_numpy(np.array(cond_image)),
            return_tensors="pt",
        ).pixel_values

    def fix_channel_not_last(self, x):
        assert len(x.shape) == 3, "Expected 3D tensor"
        if x.shape[0] < x.shape[-1]:
            return np.transpose(x, (1, 2, 0))

    def prep_sparse_dm(self, sparse_dms, interpolation_mode):
        if interpolation_mode is None:
            return sparse_dms
        if len(sparse_dms.shape) == 4:
            cond_images = []
            for sdm in sparse_dms:
                cond_images.append(self._prep_sparse_dm(sdm, interpolation_mode))
            cond_images = torch.stack(cond_images, dim=0)
        else:
            cond_images = self._prep_sparse_dm(sparse_dms, interpolation_mode)
        return cond_images

    def _prep_sparse_dm(self, sdm, interpolation_mode):
        if interpolation_mode is None:
            return sdm
        return torch.from_numpy(
            data_utils.infill_sparse_depth(sdm.numpy())[0]
            if interpolation_mode == "infill"
            else data_utils.interpolate_sparse_depth(
                sdm.squeeze().numpy(), do_multiscale=True
            )
        ).unsqueeze(0)
