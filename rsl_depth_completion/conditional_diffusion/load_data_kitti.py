import argparse
import os

import torch

# from kbnet import data_utils
# import utils as data_utils
import yaml
from rsl_depth_completion.conditional_diffusion import utils as data_utils
from rsl_depth_completion.conditional_diffusion.img_utils import center_crop
from rsl_depth_completion.conditional_diffusion.load_data_base import BaseDMDataset
from rsl_depth_completion.data.kitti.kitti_dataset import CustomKittiDCDataset


class KITTIDMDataset(CustomKittiDCDataset, BaseDMDataset):
    def __init__(
        self,
        cfg,
        *args,
        **kwargs,
    ):
        self.kitti_kwargs = self.load_base_ds(cfg)
        self.sdm_interpolation_mode = cfg.sdm_interpolation_mode
        self.input_img_size = cfg.input_img_size

        CustomKittiDCDataset.__init__(self, *args, **kwargs, **self.kitti_kwargs)
        
        eval_batch = None
        if os.path.exists("eval_batch.pt"):
            total_eval_batch = torch.load("eval_batch.pt")
            if cfg.input_res in total_eval_batch:
                eval_batch = {k:v[:cfg.batch_size] for k,v in torch.load("eval_batch.pt")[cfg.input_res].items()}
            else:
                print(f"eval_batch.pt does not contain {cfg.input_res}")
        BaseDMDataset.__init__(self, *args, eval_batch=eval_batch, **kwargs, **self.kitti_kwargs)

    def load_base_ds(self, cfg):
        ds_config_str = open(
            f"{cfg.path_to_project_dir}/rsl_depth_completion/configs/data/kitti_custom.yaml"
        ).read()
        ds_config_str = ds_config_str.replace("${data_dir}", cfg.base_kitti_dataset_dir)
        ds_config = argparse.Namespace(**yaml.safe_load(ds_config_str)["ds_config"])
        ds_config.use_pose = "photo" in ds_config.train_mode
        ds_config.result = ds_config.result_dir
        ds_config.use_rgb = ("rgb" in ds_config.input) or ds_config.use_pose
        ds_config.use_d = "d" in ds_config.input
        ds_config.use_g = "g" in ds_config.input

        ds_config.data_dir = cfg.base_kitti_dataset_dir
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
        kitti_kwargs = {
            "image_paths": sorted(val_image_paths),
            "sparse_depth_paths": sorted(val_sparse_depth_paths),
            "intrinsics_paths": sorted(val_intrinsics_paths),
            "ground_truth_paths": sorted(val_ground_truth_paths),
            "ds_config": ds_config,
        }
        return kitti_kwargs

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        if self.do_crop:
            items["d"] = center_crop(items["d"], crop_size=self.input_img_size)
            items["img"] = center_crop(items["img"], crop_size=self.input_img_size)
        sparse_dm = items["d"]
        sparse_dm /= self.max_depth

        interpolated_sparse_dm = torch.from_numpy(
            data_utils.infill_sparse_depth(sparse_dm.numpy())[0]
            if self.sdm_interpolation_mode == "infill"
            else data_utils.interpolate_sparse_depth(
                sparse_dm.squeeze().numpy(), do_multiscale=True
            )
        ).unsqueeze(2)

        rgb_image = items["img"]

        sample = {
            "input_img": interpolated_sparse_dm.detach().numpy(),
            "sdm": (sparse_dm).detach(),
        }
        sample = self.extend_sample(sparse_dm, rgb_image, sample)

        return sample
