import cv2
import numpy as np
import torch
from utils import load_extractors


class BaseDMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        use_rgb_as_text_embed=False,
        use_rgb_as_cond_image=True,
        use_cond_image=True,
        use_text_embed=True,
        do_crop=True,
        include_sdm_and_rgb_in_sample=True,
        *args,
        **kwargs,
    ):
        self.use_rgb_as_text_embed = use_rgb_as_text_embed
        self.use_rgb_as_cond_image = use_rgb_as_cond_image
        self.use_cond_image = use_cond_image
        self.use_text_embed = use_text_embed
        self.extractor_model, self.extractor_processor = load_extractors()
        self.max_depth = 80
        self.do_crop = do_crop
        self.include_sdm_and_rgb_in_sample = include_sdm_and_rgb_in_sample

    def extend_sample(self, sparse_dm, rgb_image, sample):
        extension = {}
        if self.use_cond_image:
            if self.use_rgb_as_cond_image:
                extension["cond_image"] = (rgb_image / 255).detach()
            else:
                extension["cond_image"] = (sparse_dm / self.max_depth).detach()
            extension["cond_image"] *= 2
            extension["cond_image"] -= 1

        if self.use_text_embed:
            if self.use_rgb_as_text_embed:
                rgb_pixel_values = self.extract_img_features(rgb_image)
                rgb_embed = self.extractor_model.get_image_features(
                    pixel_values=rgb_pixel_values
                )

                extension["text_embed"] = rgb_embed.detach()
            else:
                sdm_pixel_values = self.extract_img_features(
                    cv2.cvtColor(sparse_dm.squeeze().numpy(), cv2.COLOR_GRAY2RGB)
                )
                sdm_embed = self.extractor_model.get_image_features(
                    pixel_values=sdm_pixel_values
                )
                extension["text_embed"] = sdm_embed.detach()
        if self.include_sdm_and_rgb_in_sample:
            extension["sdm"] = (sparse_dm).detach()
            extension["rgb"] = (rgb_image / 255).detach()
        return {**sample, **extension}

    def extract_img_features(self, cond_image):
        return self.extractor_processor(
            images=torch.from_numpy(np.array(cond_image)),
            return_tensors="pt",
        ).pixel_values
