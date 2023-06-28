import torch
from datasets import load_dataset
from rsl_depth_completion.conditional_diffusion.img_utils import center_crop
from rsl_depth_completion.conditional_diffusion.load_data_base import BaseDMDataset


def mnist_transforms(examples, transform=None):
    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples["input_img"]
    ]
    del examples["input_img"]

    return examples


class MNISTDMDataset(BaseDMDataset):
    def __init__(
        self,
        img_transform,
        cfg,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        dataset = load_dataset("fashion_mnist")
        train_dataset = dataset["train"]

        self.train_dataset = train_dataset.with_transform(img_transform).remove_columns(
            "label"
        )
        self.rgb_image = torch.load("./rgb_image.pt")
        self.sparse_dm = torch.load("./sparse_dm.pt") / self.max_depth
        self.input_img_size = cfg.input_img_size

        if self.do_crop:
            self.rgb_image = center_crop(self.rgb_image, crop_size=self.input_img_size)
            self.sparse_dm = center_crop(self.sparse_dm, crop_size=self.input_img_size)

    def __getitem__(self, idx):
        mnist_img = self.train_dataset[idx]

        sample = {
            "input_img": mnist_img["pixel_values"],
        }

        extension = self.extend_sample(self.sparse_dm, self.rgb_image)
        sample.update(extension)
        return sample

    def __len__(self):
        return len(self.train_dataset)
