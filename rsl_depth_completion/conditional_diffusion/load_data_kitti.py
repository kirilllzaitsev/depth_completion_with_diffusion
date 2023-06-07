import argparse
import os

import torch

# from kbnet import data_utils
# import utils as data_utils
import yaml
from rsl_depth_completion.conditional_diffusion import utils as data_utils
from rsl_depth_completion.conditional_diffusion.config import cfg
from rsl_depth_completion.conditional_diffusion.img_utils import center_crop, resize
from rsl_depth_completion.conditional_diffusion.load_data_base import BaseDMDataset
from rsl_depth_completion.data.kitti.kitti_dataset import CustomKittiDCDataset


class KITTIDMDataset(CustomKittiDCDataset, BaseDMDataset):
    def __init__(
        self,
        cfg: cfg,
        *args,
        **kwargs,
    ):
        self.kitti_kwargs = self.load_base_ds(cfg)
        self.do_crop = cfg.do_crop
        self.input_img_size = cfg.input_img_size
        self.max_input_img_size = cfg.max_input_img_size

        CustomKittiDCDataset.__init__(self, *args, **kwargs, **self.kitti_kwargs)

        eval_batch = None
        eval_batch_path = cfg.eval_batch_path
        if os.path.exists(eval_batch_path):
            total_eval_batch = torch.load(eval_batch_path)
            if cfg.input_res in total_eval_batch:
                eval_batch = {
                    k: v[: cfg.batch_size]
                    for k, v in torch.load(eval_batch_path)[cfg.input_res].items()
                }
            else:
                print(f"{eval_batch_path} does not contain {cfg.input_res}")

        assert len(cfg.unets_output_res) <= 2
        base_unet_res = cfg.unets_output_res[0]
        target_lowres_img_size = None
        if cfg.input_res > base_unet_res:
            target_lowres_img_size = (base_unet_res, base_unet_res)

        BaseDMDataset.__init__(
            self,
            cfg=cfg,
            *args,
            eval_batch=eval_batch,
            max_depth=cfg.max_depth,
            cond_img_sdm_interpolation_mode=cfg.cond_img_sdm_interpolation_mode,
            input_img_sdm_interpolation_mode=cfg.input_img_sdm_interpolation_mode,
            target_lowres_img_size=target_lowres_img_size,
            **kwargs,
            **self.kitti_kwargs,
        )

    def load_base_ds(self, cfg):
        ds_config_str = open(
            f"{cfg.path_to_project_dir}/rsl_depth_completion/configs/data/kitti_custom.yaml"
        ).read()
        ds_config_str = ds_config_str.replace("${data_dir}", cfg.base_kitti_dataset_dir)
        ds_config = yaml.safe_load(ds_config_str)["ds_config"]

        path_to_data_dir = cfg.base_kitti_dataset_dir
        image_paths = data_utils.read_paths(
            ds_config[f'{ds_config["split"]}_image_path'],
            path_to_data_dir=path_to_data_dir,
        )
        sparse_depth_paths = data_utils.read_paths(
            ds_config[f'{ds_config["split"]}_sparse_depth_path'],
            path_to_data_dir=path_to_data_dir,
        )
        intrinsics_paths = data_utils.read_paths(
            ds_config[f'{ds_config["split"]}_intrinsics_path'],
            path_to_data_dir=path_to_data_dir,
        )
        ground_truth_paths = data_utils.read_paths(
            ds_config[f'{ds_config["split"]}_ground_truth_path'],
            path_to_data_dir=path_to_data_dir,
        )
        ds_config = argparse.Namespace(**ds_config)
        ds_config.data_dir = cfg.base_kitti_dataset_dir
        ds_config.use_pose = "photo" in ds_config.train_mode
        ds_config.result = ds_config.result_dir
        ds_config.use_rgb = ("rgb" in ds_config.input) or ds_config.use_pose
        ds_config.use_d = "d" in ds_config.input
        ds_config.use_g = "g" in ds_config.input
        kitti_kwargs = {
            "image_paths": sorted(image_paths),
            "sparse_depth_paths": sorted(sparse_depth_paths),
            "intrinsics_paths": sorted(intrinsics_paths),
            "ground_truth_paths": sorted(ground_truth_paths),
            "ds_config": ds_config,
        }
        return kitti_kwargs

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        if self.do_crop:
            items["gt"] = self.prep_img(items["gt"])
            items["d"] = self.prep_img(items["d"])
            items["img"] = self.prep_img(items["img"])
            if "adj_imgs" in items:
                adj_imgs = []
                for i, img in enumerate(items["adj_imgs"]):
                    cropped_adj_img = self.prep_img(img)
                    adj_imgs.append(cropped_adj_img)
                items["adj_imgs"] = torch.stack(adj_imgs, dim=0).float()
        sparse_dm = items["d"]
        sparse_dm /= self.max_depth

        interpolated_sparse_dm = self.prep_sparse_dm(
            sparse_dm, self.input_img_sdm_interpolation_mode
        )

        rgb_image = items["img"]

        sample = {
            "input_img": interpolated_sparse_dm.detach().numpy(),
            **{
                k: v.detach().numpy() for k, v in items.items() if k not in ["d", "img"]
            },
        }
        extension = self.extend_sample(sparse_dm, rgb_image)
        sample.update(extension)

        return sample

    def prep_img(self, x, channels_last=False):
        x = center_crop(x, crop_size=self.max_input_img_size, channels_last=channels_last)
        return x
