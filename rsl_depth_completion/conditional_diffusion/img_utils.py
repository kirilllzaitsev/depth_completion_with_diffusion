import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv


def center_crop(img, crop_size, channels_last=False):
    has_batch_dim = len(img.shape) == 4
    if channels_last:
        h, w = img.shape[1:3] if has_batch_dim else img.shape[:2]
    else:
        h, w = img.shape[2:4] if has_batch_dim else img.shape[1:]
    th, tw = crop_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    if channels_last:
        if has_batch_dim:
            return img[:, i : i + th, j : j + tw, :]
        return img[i : i + th, j : j + tw, :]
    if has_batch_dim:
        return img[:, :, i : i + th, j : j + tw]
    return img[:, i : i + th, j : j + tw]


def resize(
    img: np.ndarray, size, interpolation=tv.transforms.InterpolationMode.BICUBIC
):
    assert interpolation in [
        tv.transforms.InterpolationMode.BICUBIC,
        tv.transforms.InterpolationMode.BILINEAR,
    ]
    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(size, antialias=True, interpolation=interpolation),
        ]
    )
    img = fix_channel_not_first(img)
    resized = transform(img)
    return resized


def fix_channel_not_last(x):
    assert len(x.shape) >= 3, "Must be at least a 3D tensor"
    if len(x.shape) == 3:
        if x.shape[0] < x.shape[-1]:
            return np.transpose(x, (1, 2, 0))
    elif len(x.shape) == 4:
        if x.shape[1] < x.shape[-1]:
            return np.transpose(x, (0, 2, 3, 1))
    return x


def fix_channel_not_first(x):
    assert len(x.shape) >= 3, "Must be at least a 3D tensor"
    if len(x.shape) == 3:
        if x.shape[0] > x.shape[-1]:
            return np.transpose(x, (2, 0, 1))
    elif len(x.shape) == 4:
        if x.shape[1] > x.shape[-1]:
            return np.transpose(x, (0, 3, 1, 2))
    return x


def plot_sr_samples(samples, eval_batch, kbnet_preds=None):
    is_kbnet_available = kbnet_preds is not None
    num_cols = 3 if is_kbnet_available is None else 4
    if is_kbnet_available is None:
        kbnet_preds = [None] * len(samples)
    fig, axs = plt.subplots(2, num_cols, figsize=(6, 5))
    for i, (sample, lowres, input_img, kbnet_pred) in enumerate(
        zip(samples, eval_batch["lowres_img"], eval_batch["input_img"], kbnet_preds)
    ):
        axs[i, 0].imshow(lowres.permute(1, 2, 0).cpu().numpy())
        axs[i, 1].imshow(input_img.permute(1, 2, 0).cpu().numpy())
        axs[i, 2].imshow(sample.permute(1, 2, 0).cpu().numpy())
        if kbnet_pred is not None:
            axs[i, 3].imshow(kbnet_pred.permute(1, 2, 0).cpu().detach().numpy())
    axs[0, 0].set_title("lowres")
    axs[0, 1].set_title("cond_img")
    axs[0, 2].set_title("sample")
    if is_kbnet_available is not None:
        axs[0, 3].set_title("kbnet_pred")

    for ax in axs.flatten():
        ax.axis("off")
