"""A vanilla PyTorch Dataset (think about porting it to Lighting, though what are the benefits?)"""

import os
import random

import cv2
import numpy as np
import torch

from .kitti_utils import camera_calibration as calib
from .kitti_utils import custom_transforms as transforms
from .kitti_utils import pose_estimator as pose
from .kitti_utils import raw_data_loaders as dl
from .kitti_utils.data_logging import logger


def extract_frame_id_from_img_path(filename: str):
    head, tail = os.path.split(filename)
    number_string = tail[0 : tail.find(".")]
    number = int(number_string)
    return head, number


def get_nearby_img_path(filename: str, new_id: int):
    head, _ = os.path.split(filename)
    new_filename = os.path.join(head, f"{new_id:010d}.png")
    return new_filename


def get_adj_imgs(path: str, config):
    assert path is not None, "path is None"

    _, frame_id = extract_frame_id_from_img_path(path)
    offsets = []
    for i in range(1, config.n_adjacent + 1):
        offsets.append(i)
        offsets.append(-i)
    offsets_asc = sorted(offsets)

    adj_imgs = []
    for frame_offset in offsets_asc:
        path_near = get_nearby_img_path(path, frame_id + frame_offset)
        assert os.path.exists(path_near), f"cannot find two nearby frames for {path}"
        adj_imgs.append(dl.img_read(path_near))

    return adj_imgs


class KittiDCDataset(torch.utils.data.Dataset):
    """
    Loads the KITTI dataset for depth completion.
    One item:
        frame: (3, img_height, img_width)
        adjacent_frames: (frame-to-the-left, frame-to-the-right) with the same shape as frame
        depth: sparse_dm depth map of shape (1, img_height, img_width)

    """

    def __init__(self, config):
        self.config = config
        self.split = config.split
        self.subsplit = config.subsplit if config.subsplit is not None else "train"
        paths, transform = dl.get_paths_and_transform(self.split, self.subsplit, config)
        self.paths = paths
        self.transform = transform
        self.K = calib.load_calib()
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
        sample = self.load_sample(index)
        img, sparse_dm, dense_dm, adj_imgs = self.transform(sample)
        t_vecs = []
        r_mats = []
        if self.split == "train" and self.config.use_pose:
            for i, adj_img_ in enumerate(adj_imgs):
                success, r_mat, t_vec = self.estimate_pose(img, sparse_dm, adj_img_)
                if not success:
                    # return the same image and no motion when PnP fails
                    logger.warning(
                        f"PnP failed, returning the same image for the {i-config.n_adjacent}th adjacent frame of the sample {self.paths['img'][index]}"
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


def create_dataloader(config):
    """Creates a dataloader for the KITTI dataset"""
    dataset = KittiDCDataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Sparse-to-Dense")
    parser.add_argument(
        "-w",
        "--num_workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=11,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 11)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-c",
        "--criterion",
        metavar="LOSS",
        default="l2",
        # choices=criteria.loss_names,
        # help="loss function: | ".join(criteria.loss_names) + " (default: l2)",
    )
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="mini-batch size (default: 1)"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        metavar="LR",
        help="initial learning rate (default 1e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 0)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--data-folder",
        default="../data",
        type=str,
        metavar="PATH",
        help="data folder (default: none)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="gd",
        # choices=input_options,
        # help="input: | ".join(input_options),
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        default=34,
        help="use 16 for sparse_conv; use 18 or 34 for resnet",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="use ImageNet pre-trained weights"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="select the split to use",
    )
    parser.add_argument(
        "--subsplit",
        type=str,
        default="select",
        choices=["select", "full", "completion", "prediction"],
        help="select the subsplit in the selected split (required for val and test)",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.1, help="color jitter for images"
    )
    parser.add_argument(
        "--rank-metric",
        type=str,
        default="rmse",
        # choices=[m for m in dir(Result()) if not m.startswith("_")],
        help="metrics for which best result is sbatch_datacted",
    )
    parser.add_argument(
        "-m",
        "--train-mode",
        type=str,
        default="dense",
        choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
        help="dense | sparse | photo | sparse+photo | dense+photo",
    )
    parser.add_argument("-e", "--evaluate", default="", type=str, metavar="PATH")
    parser.add_argument(
        "--n-adjacent",
        default=1,
        type=int,
        help="range in which to take adjacent frames. 0 for no adjacent frames",
    )
    parser.add_argument("--cpu", action="store_true", help="run on cpu")
    config = parser.parse_args()
    config.use_pose = "photo" in config.train_mode
    # config.pretrained = not config.no_pretrained
    config.result = os.path.join("..", "results")
    config.use_rgb = ("rgb" in config.input) or config.use_pose
    config.use_d = "d" in config.input
    config.use_g = "g" in config.input
    print(config)
    dataloader = create_dataloader(config)
    x = next(iter(dataloader))
    print(x)
