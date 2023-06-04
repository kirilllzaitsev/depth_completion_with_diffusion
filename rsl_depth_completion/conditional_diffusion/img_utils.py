
import numpy as np
import torchvision as tv


def center_crop(img, crop_size, channels_last=False):
    if channels_last:
        h, w = img.shape[:2]
    else:
        h, w = img.shape[1:3]
    th, tw = crop_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    if channels_last:
        return img[i : i + th, j : j + tw, :]
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
