import argparse

import cv2
import numpy as np
import torch
import yaml
from config import base_kitti_dataset_dir, is_cluster, path_to_project_dir, tmpdir
from kbnet import data_utils
from rsl_depth_completion.conditional_diffusion import utils
from rsl_depth_completion.data.kitti.kitti_dataset import CustomKittiDCDataset
from torchvision import transforms
from utils import load_extractors

extractor_model, extractor_processor = load_extractors()


dataset = load_dataset("fashion_mnist")
img_size = (64, 64)
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)


def mnist_transforms(examples):
    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples["image"]
    ]
    del examples["image"]

    return examples


class DCDatasetCustom:
    def __init__(
        self,
        include_cond_image=False,
        sdm_transform=None,
        dataset=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.include_cond_image = include_cond_image
        self.default_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.sdm_transform = sdm_transform or self.default_transform
        self.max_depth = 80

        if dataset is not None:
            self.train_dataset = dataset.with_transform(
                mnist_transforms
            ).remove_columns("label")

    def __getitem__(self, idx):
        img = self.train_dataset[idx]
        img = transforms.Resize((64, 64), antialias=True)(img["pixel_values"])
        cond_image = torch.load("cond_image.pt")

        pixel_values = extractor_processor(
            images=torch.stack(
                [
                    torch.from_numpy(np.array(cond_image)),
                    torch.from_numpy(np.array(cond_image)),
                ]
            ),
            return_tensors="pt",
        ).pixel_values
        embedding = extractor_model.get_image_features(pixel_values=pixel_values)
        embedding = embedding.unsqueeze(1)
        encoding = embedding[0]
        mask = torch.ones(1).bool()

        sample = {
            "image": img.detach(),
            "encoding": encoding.detach(),
            "mask": mask.detach(),
        }
        if self.include_cond_image:
            sample["cond_image"] = cond_image.detach()
        return sample


ds = DCDatasetCustom(
    include_cond_image=True,
    dataset=dataset["train"],
)


ds_subset = torch.utils.data.Subset(
    ds,
    range(0, len(ds) // 2)
    # range(0, 5)
)
train_size = int(0.8 * len(ds_subset))
test_size = len(ds_subset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    ds_subset, [train_size, test_size]
)
if is_cluster:
    BATCH_SIZE = 16
    NUM_WORKERS = min(20, BATCH_SIZE)
else:
    BATCH_SIZE = 2
    NUM_WORKERS = 0

dl_opts = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "drop_last": True,
}
train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts, shuffle=False)
