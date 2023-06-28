"""A vanilla PyTorch Dataset (think about porting it to Lighting, though what are the benefits?)"""

import os
import random

import cv2
import numpy as np
import torch
from rsl_depth_completion.data.components import camera_calibration as calib
from rsl_depth_completion.data.components import custom_transforms as transforms
from rsl_depth_completion.data.components import pose_estimator as pose
from rsl_depth_completion.data.components import raw_data_loaders as dl
from rsl_depth_completion.data.components.adjacent_img_handler import get_adj_imgs
from rsl_depth_completion.data.components.data_logging import logger


class KittiDCDataset(torch.utils.data.Dataset):
    """
    Loads the KITTI dataset for depth completion.
    One item:
        frame: (3, img_height, img_width)
        adjacent_frames: (frame-to-the-left, frame-to-the-right) with the same shape as frame
        depth: sparse_dm depth map of shape (1, img_height, img_width)

    """

    def __init__(
        self,
        ds_config,
        path_to_depth_completion_dir,
        path_to_calib_files,
        *args,
        **kwargs,
    ):
        self.config = ds_config
        self.split = ds_config.split
        self.subsplit = (
            ds_config.subsplit if ds_config.subsplit is not None else "train"
        )
        self.paths, self.transform = dl.get_paths_and_transform(
            self.split, self.subsplit, path_to_depth_completion_dir, ds_config
        )
        self.K = calib.load_calib(path_to_calib_files)
        self.threshold_translation = 0.1

    def load_sample(self, index):
        img = (
            dl.img_read(self.paths["img"][index])
            if (
                self.paths["img"][index] is not None
                and (self.config.use_rgb or self.config.use_g)
            )
            else None
        )
        sparse_dm = (
            dl.depth_read(self.paths["d"][index])
            if (self.paths["d"][index] is not None and self.config.use_d)
            else None
        )
        dense_dm = (
            dl.depth_read(self.paths["gt"][index])
            if self.paths["gt"][index] is not None
            else None
        )
        adj_imgs = (
            get_adj_imgs(self.paths["img"][index], self.config)
            if self.split == "train" and self.config.use_pose
            else None
        )

        return {
            "img": img,
            "sparse_dm": sparse_dm,
            "dense_dm": dense_dm,
            "adj_imgs": adj_imgs,
        }

    def __getitem__(self, index):
        """
        Returns:
            A dictionary of the following items:
                img: (3, img_height, img_width)
                d: sparse_dm depth map of shape (1, img_height, img_width)
                gt: dense_dm depth map of shape (1, img_height, img_width)
                g: grayscale image of shape (1, img_height, img_width)
                adj_imgs: (config.n_adjacent, 3, img_height, img_width)
                r_mats: (config.n_adjacent, 3, 3)
                t_vecs: (config.n_adjacent, 3)

        """
        sample = self.load_sample(index)
        sample = self.transform(sample)
        img = sample["img"]
        sparse_dm = sample["sparse_dm"]
        dense_dm = sample["dense_dm"]
        adj_imgs = sample["adj_imgs"] or []

        candidates = {
            "img": img if self.config.use_rgb else None,
            "d": sparse_dm,
            "gt": dense_dm,
            "g": dl.convert_img_to_grayscale(img) if self.config.use_g else None,
        }
        items = {
            key: transforms.to_float_tensor(val)
            for key, val in candidates.items()
            if val is not None
        }

        for i in range(len(adj_imgs)):
            adj_imgs[i] = transforms.to_float_tensor(adj_imgs[i])
        items["adj_imgs"] = torch.stack(adj_imgs, dim=0).float()
        return items

    def estimate_pose(self, img, sparse_dm, adj_imgs):
        success, r_vec, t_vec = pose.get_pose_pnp(img, adj_imgs, sparse_dm, self.K)
        # discard if translation is too small
        success = success and np.linalg.norm(t_vec) > self.threshold_translation
        if success:
            r_mat, _ = cv2.Rodrigues(r_vec)
        else:
            t_vec = np.zeros((3, 1))
            r_mat = np.eye(3)
        return success, r_mat, t_vec

    def __len__(self):
        return len(self.paths["img"])


class CustomKittiDCDataset(KittiDCDataset):
    """
    Loads the KITTI dataset for depth completion.
    One item:
        frame: (3, img_height, img_width)
        adjacent_frames: (frame-to-the-left, frame-to-the-right) with the same shape as frame
        depth: sparse_dm depth map of shape (1, img_height, img_width)

    """

    def __init__(
        self,
        image_paths,
        sparse_depth_paths,
        intrinsics_paths,
        ds_config,
        ground_truth_paths=None,
        transform=None,
        *args,
        **kwargs,
    ):
        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths
        self.ground_truth_paths = ground_truth_paths
        self.transform = transform
        self.paths = self.get_input_paths()

        self.config = ds_config
        self.split = ds_config.split
        self.subsplit = (
            ds_config.subsplit if ds_config.subsplit is not None else "train"
        )
        self.transform = self.get_data_transform(ds_config)
        self.K = self.load_intrinsic_calibration_matrix()
        self.threshold_translation = 0.1

        # self.validate_paths()

    def load_intrinsic_calibration_matrix(
        self,
    ):
        """As intrinsics is used to estimate [R|t], could get it at __getitem__ based on index, if
        there are multiple items in the intrinsics_paths."""
        return np.load(self.intrinsics_paths[0]).astype(np.float32)

    def validate_paths(self):
        assert len(self.image_paths) == len(self.sparse_depth_paths)
        if self.ground_truth_paths is not None:
            assert len(self.image_paths) == len(self.ground_truth_paths)

    def get_input_paths(
        self,
    ):
        return {
            "img": self.image_paths,
            "d": self.sparse_depth_paths,
            "gt": self.ground_truth_paths,
        }

    def get_data_transform(self, ds_config):
        return dl.get_data_transform(ds_config.split, ds_config.subsplit, ds_config)

    def __getitem__(self, index):
        items = super().__getitem__(index)

        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)
        items["intrinsics"] = torch.from_numpy(intrinsics)
        return items


if __name__ == "__main__":
    import argparse

    import yaml
    from kbnet import data_utils

    ds_config = argparse.Namespace(
        **yaml.safe_load(open("../../../configs/data/kitti_custom.yaml"))["ds_config"]
    )
    ds_config.use_pose = "photo" in ds_config.train_mode
    ds_config.result = ds_config.result_dir
    ds_config.use_rgb = ("rgb" in ds_config.input) or ds_config.use_pose
    ds_config.use_d = "d" in ds_config.input
    ds_config.use_g = "g" in ds_config.input
    val_image_paths = data_utils.read_paths(ds_config.val_image_path)
    val_sparse_depth_paths = data_utils.read_paths(ds_config.val_sparse_depth_path)
    val_intrinsics_paths = data_utils.read_paths(ds_config.val_intrinsics_path)
    val_ground_truth_paths = data_utils.read_paths(ds_config.val_ground_truth_path)
    ds = CustomKittiDCDataset(
        val_image_paths,
        val_sparse_depth_paths,
        val_intrinsics_paths,
        ds_config,
        val_ground_truth_paths,
    )
    x = ds[0]
    print(x)
