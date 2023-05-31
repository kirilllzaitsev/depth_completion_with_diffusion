import numpy as np

from . import custom_transforms as transforms
from .data_logging import logger as logging

oheight, owidth = 352, 1216


class Transform:
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        return self.transform(sample)

    def transform(self, sample):
        raise NotImplementedError


class TrainTransform(Transform):
    def transform(self, sample):
        return train_transform(sample, self.config)


class ValTransform(Transform):
    def transform(self, sample):
        return val_transform(sample, self.config)


class DummyTransform(Transform):
    def transform(self, sample):
        return sample


def train_transform(sample, config):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    img, sparse_dm, dense_dm, adj_imgs = (
        sample["img"],
        sample["sparse_dm"],
        sample["dense_dm"],
        sample["adj_imgs"],
    )
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose(
        [
            # transforms.Rotate(angle),
            # transforms.Resize(s),
            transforms.BottomCrop((oheight, owidth)),
            transforms.HorizontalFlip(do_flip),
        ]
    )
    if sparse_dm is not None:
        sparse_dm = transform_geometric(sparse_dm)
    dense_dm = transform_geometric(dense_dm)
    if img is not None:
        brightness = np.random.uniform(max(0, 1 - config.jitter), 1 + config.jitter)
        contrast = np.random.uniform(max(0, 1 - config.jitter), 1 + config.jitter)
        saturation = np.random.uniform(max(0, 1 - config.jitter), 1 + config.jitter)
        transform_rgb = transforms.Compose(
            [
                transforms.ColorJitter(brightness, contrast, saturation, 0),
                transform_geometric,
            ]
        )
        img = transform_rgb(img)
        if adj_imgs is not None:
            for i in range(len(adj_imgs)):
                adj_imgs[i] = transform_rgb(adj_imgs[i])
    # sparse_dm = drop_depth_measurements(sparse_dm, 0.9)

    result = {
        "img": img,
        "sparse_dm": sparse_dm,
        "dense_dm": dense_dm,
        "adj_imgs": adj_imgs,
    }

    return result


def val_transform(sample, config):
    img, sparse_dm, dense_dm, adj_imgs = (
        sample["img"],
        sample["sparse_dm"],
        sample["dense_dm"],
        sample["adj_imgs"],
    )
    transform = transforms.Compose(
        [
            transforms.BottomCrop((oheight, owidth)),
        ]
    )
    if img is not None:
        img = transform(img)
    if sparse_dm is not None:
        sparse_dm = transform(sparse_dm)
    if dense_dm is not None:
        dense_dm = transform(dense_dm)
    if adj_imgs is not None:
        for i in range(len(adj_imgs)):
            adj_imgs[i] = transform(adj_imgs[i])
    result = {
        "img": img,
        "sparse_dm": sparse_dm,
        "dense_dm": dense_dm,
        "adj_imgs": adj_imgs,
    }

    return result
