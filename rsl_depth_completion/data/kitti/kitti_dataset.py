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
        img, sparse_dm, dense_dm, adj_imgs = self.transform(sample)
        t_vecs = []
        r_mats = []
        if self.split == "train" and self.config.use_pose:
            for i, adj_img in enumerate(adj_imgs):
                success, r_mat, t_vec = self.estimate_pose(img, sparse_dm, adj_img)
                if not success:
                    # return the same image and no motion when PnP fails
                    logger.warning(
                        f"PnP failed, returning the same image for the {i-self.config.n_adjacent}th adjacent frame of the sample {self.paths['img'][index]}"
                    )
                    adj_imgs[i] = img
                t_vecs.append(t_vec)
                r_mats.append(r_mat)

        for i in range(len(adj_imgs)):
            adj_imgs[i] = transforms.to_float_tensor(adj_imgs[i])
            t_vecs[i] = transforms.to_float_tensor(t_vecs[i])
            r_mats[i] = transforms.to_float_tensor(r_mats[i])

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
        candidates["adj_imgs"] = torch.stack(adj_imgs, dim=0).float()
        candidates["r_mats"] = torch.stack(r_mats, dim=0).float()
        candidates["t_vecs"] = torch.stack(t_vecs, dim=0).float()

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
        return len(self.paths["gt"])
