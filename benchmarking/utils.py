import datetime as dt
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from rsl_depth_completion.data.components.raw_data_loaders import depth_read, img_read


def log_imgs(name, imgs):
    tf.summary.image(
        name,
        np.array(imgs).squeeze(axis=1),
        max_outputs=len(imgs),
        step=0,
    )


def log_errors_per_img(errors_per_img, img_name, tb_name, model_names):
    for model_name in model_names:
        tf.summary.scalar(
            f"{tb_name}/metrics/rmse/{model_name}",
            errors_per_img[img_name]["metrics"]["rmse"][model_name],
            step=0,
        )
        tf.summary.scalar(
            f"{tb_name}/metrics/mae/{model_name}",
            errors_per_img[img_name]["metrics"]["mae"][model_name],
            step=0,
        )
        tf.summary.text(
            f"{tb_name}/meta/masks_with_no_prediction/{model_name}",
            ",".join(errors_per_img[img_name]["masks_with_no_prediction"][model_name]),
            step=0,
        )
    tf.summary.scalar(
        f"{tb_name}/meta/num_objects",
        errors_per_img[img_name]["num_objects"],
        step=0,
    )
    tf.summary.text(
        f"{tb_name}/meta/objects",
        ",".join(errors_per_img[img_name]["objects"]),
        step=0,
    )
    tf.summary.text(
        f"{tb_name}/meta/masks_with_no_gt",
        ",".join(errors_per_img[img_name]["masks_with_no_gt"]),
        step=0,
    )


def log_errors_per_class(errors_per_class, obj_class_no_ext, model_names):
    tf.summary.text(
        f"{obj_class_no_ext}/meta/imgs_with_no_gt",
        ",".join(errors_per_class[obj_class_no_ext]["imgs_with_no_gt"]),
        step=0,
    )
    for model_name in model_names:
        tf.summary.text(
            f"{obj_class_no_ext}/meta/imgs_with_no_prediction",
            ",".join(errors_per_class[obj_class_no_ext]["imgs_with_no_prediction"][model_name]),
            step=0,
        )
        tf.summary.scalar(
            f"{obj_class_no_ext}/metrics/rmse/{model_name}",
            errors_per_class[obj_class_no_ext]["metrics"]["rmse"][model_name],
            step=0,
        )
        tf.summary.scalar(
            f"{obj_class_no_ext}/metrics/mae/{model_name}",
            errors_per_class[obj_class_no_ext]["metrics"]["mae"][model_name],
            step=0,
        )


def extract_sample_name_from_full_img_paths(img_paths):
    return [
        img_path.rsplit("_")[-3] if len(img_path) > 10 else img_path
        for img_path in img_paths
    ]


def plot_full_results_for_all_models(
    model_names, pred_dms, img, title=None, save_path=None
):
    nrows = 4
    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(12, 10),
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
    )
    pred_depth_start_idx = 1
    for model_idx, preds in enumerate(pred_dms):
        idx = pred_depth_start_idx + model_idx
        axs[idx].imshow(preds / 255)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        axs[idx].set_ylabel(f"$\\hat{{D}}$ {model_names[model_idx]}")

    axs[0].set_title(title)
    axs[0].imshow((img) / 255)
    axs[0].set_ylabel(f"RGB")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

    return fig


def plot_masked_results_for_all_models(
    model_names,
    pred_dms,
    img,
    mask,
    obj_class,
    title=None,
    save_path=None,
):
    nrows = 4
    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(12, 10),
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
    )
    pred_depth_start_idx = 1
    for model_idx, preds in enumerate(pred_dms):
        idx = pred_depth_start_idx + model_idx
        axs[idx].imshow((preds * mask) / 255)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        axs[idx].set_ylabel(f"$\\hat{{D}}$ {model_names[model_idx]}")

    axs[0].set_title(title)

    axs[0].imshow((img * mask[:, :, np.newaxis]) / 255)
    axs[0].set_ylabel(f"RGB '{obj_class}'")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # plt.tight_layout(pad=0)
    plt.margins(0)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
    fig.tight_layout(h_pad=10)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

    return fig


def plot_results_per_model(x, gt=None, masked_class=None, save_path=None, title=None):
    nrows = 7
    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(12, 10),
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
    )
    pred_depth_axs_idx = 1
    axs[pred_depth_axs_idx].imshow(x["preds"] / 255)
    axs[pred_depth_axs_idx + 3].imshow((x["preds"] * x["mask"]) / 255)
    for idx in range(nrows):
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])

    if gt is not None:
        axs[pred_depth_axs_idx + 4].imshow(
            (((gt - x["preds"]) * x["mask"]) ** 2).clip(max=255) / 255
        )
        axs[pred_depth_axs_idx + 5].imshow(np.abs(gt - x["preds"]) * x["mask"] / 255)

        axs[pred_depth_axs_idx + 4].set_xticks([])
        axs[pred_depth_axs_idx + 4].set_yticks([])
        axs[pred_depth_axs_idx + 5].set_xticks([])
        axs[pred_depth_axs_idx + 5].set_yticks([])

    axs[0].set_title(title)
    axs[pred_depth_axs_idx - 1].imshow(x["img"])
    axs[pred_depth_axs_idx - 1].set_xticks([])
    axs[pred_depth_axs_idx - 1].set_yticks([])
    axs[pred_depth_axs_idx + 2].imshow(gt * x["mask"] / 255)
    axs[pred_depth_axs_idx + 2].set_xticks([])
    axs[pred_depth_axs_idx + 2].set_yticks([])
    axs[pred_depth_axs_idx + 1].imshow((x["img"] * x["mask"][:, :, np.newaxis]) / 255)
    axs[pred_depth_axs_idx + 1].set_xticks([])
    axs[pred_depth_axs_idx + 1].set_yticks([])

    plt.margins(0)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
    fig.tight_layout(h_pad=10)

    axs[pred_depth_axs_idx - 1].set_ylabel("RGB")
    axs[pred_depth_axs_idx + 2].set_ylabel(f"GT '{masked_class}'")
    axs[pred_depth_axs_idx].set_ylabel("$\\hat{D}$")
    axs[pred_depth_axs_idx + 1].set_ylabel(f"RGB '{masked_class}'")
    axs[pred_depth_axs_idx + 3].set_ylabel(f"$\\hat{{D}}$ '{masked_class}'")
    if gt is not None:
        axs[pred_depth_axs_idx + 4].set_ylabel("MSE")
        axs[pred_depth_axs_idx + 5].set_ylabel("MAE")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
