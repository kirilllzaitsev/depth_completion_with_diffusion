import glob
import logging
import os
import typing as t

import numpy as np
from PIL import Image

from .data_logging import logger as logging
from .data_paths_handlers import SplitDataPathsHandler
from .data_transforms import DummyTransform, TrainTransform, Transform, ValTransform


def img_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename).convert("RGB")
    # img_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    img_png = np.array(img_file, dtype="uint8")  # in the range [0,255]
    img_file.close()
    return img_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, "np.max(depth_png)={}, path={}".format(
        np.max(depth_png), filename
    )

    depth = depth_png.astype(float) / 256.0
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


def get_input_paths(split, subsplit, kitti_depth_completion_data_dir, config) -> dict:
    assert config.use_d or config.use_img or config.use_g, "no proper input selected"
    data_paths_handler = SplitDataPathsHandler(
        split, subsplit, kitti_depth_completion_data_dir
    )

    paths_img, paths_d, paths_gt = (
        data_paths_handler.paths_img,
        data_paths_handler.paths_d,
        data_paths_handler.paths_gt,
    )
    validate_paths(config, paths_img, paths_d, paths_gt)

    paths = {"img": paths_img, "d": paths_d, "gt": paths_gt}
    return paths


def get_data_transform(split, subsplit, config) -> Transform:
    if split == "train":
        transform = TrainTransform(config)
    elif split == "val":
        if subsplit == "full":
            transform = ValTransform(config)
        elif subsplit == "select":
            transform = DummyTransform(config)
        else:
            raise ValueError("Unrecognized val subsplit " + str(subsplit))
    elif split == "test":
        if subsplit in ["completion", "prediction", "custom"]:
            transform = DummyTransform(config)
        else:
            raise ValueError("Unrecognized test subsplit " + str(subsplit))
    else:
        raise ValueError("Unrecognized split " + str(split))
    return transform


def validate_paths(
    config, paths_img: list[str], paths_d: list[str], paths_gt: list[str]
):
    if len(paths_d) == 0 and len(paths_img) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images"))
    if len(paths_d) == 0 and config.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_img) == 0 and config.use_img:
        raise (RuntimeError("Requested img images but none was found"))
    if len(paths_img) == 0 and config.use_g:
        raise (RuntimeError("Requested gray images but no img was found"))
    if len(paths_img) != len(paths_d) or len(paths_img) != len(paths_gt):
        logging.error(f"{len(paths_img)=}\t{len(paths_d)=}\t{len(paths_gt)=}")
        raise (RuntimeError("Produced different sizes for datasets"))


def convert_img_to_grayscale(img):
    grayscale_img = np.array(Image.fromarray(img).convert("L"))
    grayscale_img = np.expand_dims(grayscale_img, -1)
    return grayscale_img
