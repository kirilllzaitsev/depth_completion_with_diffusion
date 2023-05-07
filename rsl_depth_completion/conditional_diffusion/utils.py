import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ip_basic import depth_map_utils
from scipy.interpolate import griddata


def load_extractors():
    from transformers import CLIPModel, CLIPProcessor

    extractor_model_ref = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(extractor_model_ref)
    processor = CLIPProcessor.from_pretrained(extractor_model_ref)
    return model, processor


def get_model_params(parameters_dir):
    params = {}
    for file in os.listdir(f"{parameters_dir}"):
        if file.endswith(".json"):
            with open(f"{parameters_dir}/{file}") as f:
                params[os.path.splitext(file)[0]] = json.load(f)

    return params


def infill_sparse_depth(sdm, method="nearest"):
    # Find the indices of the missing values
    missing_indices = np.argwhere(sdm == 0)

    # Find the indices of the valid values
    valid_indices = np.argwhere(~(sdm == 0))

    # Extract the valid depth values and their corresponding indices
    valid_depth = sdm[~(sdm == 0)]
    valid_coords = valid_indices.astype(float)

    # Perform nearest neighbor interpolation to estimate the missing depth values
    interp_depth = griddata(valid_coords, valid_depth, missing_indices, method=method)

    # Replace the missing values in the original depth map with the interpolated values
    near_img = sdm.copy()
    near_img[sdm == 0] = interp_depth
    return near_img


def interpolate_sparse_depth(sdm, do_multiscale=False, *args, **kwargs):
    """See depth_map_utils.fill_in_fast"""
    if do_multiscale:
        ddm, _ = depth_map_utils.fill_in_multiscale(
            sdm.astype("float32"), *args, **kwargs
        )
    else:
        ddm = depth_map_utils.fill_in_fast(sdm.astype("float32"), *args, **kwargs)
    return ddm


def overlay_sparse_depth_and_rgb(rgb, sdm):
    sdm = sdm.squeeze().numpy()
    rgb = rgb.squeeze().numpy().transpose(1, 2, 0)
    sdm = interpolate_sparse_depth(sdm, max_depth=80)
    sdm = cv2.cvtColor(sdm, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(rgb, 0.5, sdm, 0.5, 0)


def plot_sample(x, items):
    fix, ax = plt.subplots(len(items) // 2, 2, figsize=(15, 10))
    for idx, item in enumerate(items):
        img = x[item].numpy().transpose(1, 2, 0)

        ax_idx = idx // 2, idx % 2
        ax[ax_idx].imshow(img, cmap="gray")
        ax[ax_idx].set_title(
            item,
        )
    plt.show()


def plot_source_and_reconstruction(src, rec):
    concatenated_img_and_denoised = torch.concatenate([src, rec], dim=2)
    plt.imshow(concatenated_img_and_denoised.permute(1, 2, 0).cpu().detach().numpy())


def read_paths(filepath, path_to_data_dir=None):
    """
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    """

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip("\n")
            # If there was nothing to read
            if path == "":
                break
            if path_to_data_dir is not None:
                path = path_to_data_dir + "/" + path
            path_list.append(path)

    return path_list


def log_params_to_exp(experiment, params: dict, prefix: str):
    experiment.log_parameters({f"{prefix}/{k}": v for k, v in params.items()})
