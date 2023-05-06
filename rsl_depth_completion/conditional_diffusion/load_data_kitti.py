import argparse

import numpy as np
import torch
import yaml
from config import base_kitti_dataset_dir, is_cluster, path_to_project_dir, tmpdir
from kbnet import data_utils

ds_config_str = open(
    f"{path_to_project_dir}/rsl_depth_completion/configs/data/kitti_custom.yaml"
).read()
ds_config_str = ds_config_str.replace("${data_dir}", base_kitti_dataset_dir)
ds_config = argparse.Namespace(**yaml.safe_load(ds_config_str)["ds_config"])
ds_config.use_pose = "photo" in ds_config.train_mode
ds_config.result = ds_config.result_dir
ds_config.use_rgb = ("rgb" in ds_config.input) or ds_config.use_pose
ds_config.use_d = "d" in ds_config.input
ds_config.use_g = "g" in ds_config.input

ds_config.data_dir = base_kitti_dataset_dir
path_to_data_dir = ds_config.data_dir
val_image_paths = data_utils.read_paths(
    ds_config.val_image_path, path_to_data_dir=path_to_data_dir
)
val_sparse_depth_paths = data_utils.read_paths(
    ds_config.val_sparse_depth_path, path_to_data_dir=path_to_data_dir
)
val_intrinsics_paths = data_utils.read_paths(
    ds_config.val_intrinsics_path, path_to_data_dir=path_to_data_dir
)
val_ground_truth_paths = data_utils.read_paths(
    ds_config.val_ground_truth_path, path_to_data_dir=path_to_data_dir
)

import cv2
from rsl_depth_completion.conditional_diffusion import utils
from rsl_depth_completion.data.kitti.kitti_dataset import CustomKittiDCDataset
from torchvision import transforms
from utils import load_extractors

extractor_model, extractor_processor = load_extractors()


class DMDataset(CustomKittiDCDataset):
    def __init__(
        self,
        include_cond_image=False,
        sdm_transform=None,
        do_crop=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.include_cond_image = include_cond_image
        self.default_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.sdm_transform = sdm_transform or self.default_transform
        self.max_depth = 80
        self.do_crop = do_crop

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        if self.do_crop:
            items["d"] = items["d"][:, 50 : 50 + 256, 400 : 400 + 256]
            items["img"] = items["img"][:, 50 : 50 + 256, 400 : 400 + 256]
        sparse_dm = items["d"]
        sparse_dm /= self.max_depth

        interpolated_sparse_dm = torch.from_numpy(
            # utils.infill_sparse_depth(sparse_dm.numpy())
            utils.interpolate_sparse_depth(
                sparse_dm.squeeze().numpy(), do_multiscale=True
            )
        ).unsqueeze(0)

        rgb_image = items["img"]

        rgb_pixel_values = self.extract_img_features(rgb_image)
        sdm_pixel_values = self.extract_img_features(
            cv2.cvtColor(sparse_dm.squeeze().numpy(), cv2.COLOR_GRAY2RGB)
        )
        rgb_embed = extractor_model.get_image_features(pixel_values=rgb_pixel_values)
        sdm_embed = extractor_model.get_image_features(pixel_values=sdm_pixel_values)

        sample = {
            "perturbed_sdm": interpolated_sparse_dm.detach(),
            "rgb_embed": rgb_embed.detach(),
            "sdm_embed": sdm_embed.detach(),
            "rgb_image": (rgb_image / 255).detach(),
            "sparse_dm": (sparse_dm).detach(),
        }
        return sample

    def extract_img_features(self, cond_image):
        return extractor_processor(
            images=torch.stack(
                [
                    torch.from_numpy(np.array(cond_image)),
                ]
            ),
            return_tensors="pt",
        ).pixel_values


ds = DMDataset(
    ds_config=ds_config,
    image_paths=val_image_paths,
    sparse_depth_paths=val_sparse_depth_paths,
    intrinsics_paths=val_intrinsics_paths,
    ground_truth_paths=val_ground_truth_paths,
    include_cond_image=True,
    do_crop=True,
)

ds_subset = torch.utils.data.Subset(
    ds,
    range(0, len(ds) // 2)
    # range(0, 5)
)
train_size = int(0.8 * len(ds_subset))
test_size = len(ds_subset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    ds_subset, [train_size, test_size]
)
if is_cluster:
    BATCH_SIZE = 16
    NUM_WORKERS = min(20, BATCH_SIZE)
else:
    BATCH_SIZE = 2
    NUM_WORKERS = 0

dl_opts = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "drop_last": True,
}
train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts, shuffle=False)
