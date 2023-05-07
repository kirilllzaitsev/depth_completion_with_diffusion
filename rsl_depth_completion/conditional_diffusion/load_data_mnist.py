import torch
from datasets import load_dataset
from rsl_depth_completion.conditional_diffusion.load_data_base import BaseDMDataset
from utils import load_extractors

dataset = load_dataset("fashion_mnist")
extractor_model, extractor_processor = load_extractors()


def mnist_transforms(examples, transform=None):
    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples["image"]
    ]
    del examples["image"]

    return examples


class MNISTDMDataset(BaseDMDataset):
    def __init__(
        self,
        img_transform,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        train_dataset = dataset["train"]

        self.train_dataset = train_dataset.with_transform(img_transform).remove_columns(
            "label"
        )
        self.rgb_image = torch.load("./rgb_image.pt")
        self.sparse_dm = torch.load("./sparse_dm.pt") / self.max_depth

        if self.do_crop:
            self.rgb_image = self.rgb_image[:, 50 : 50 + 256, 400 : 400 + 256]
            self.sparse_dm = self.sparse_dm[:, 50 : 50 + 256, 400 : 400 + 256]

    def __getitem__(self, idx):
        mnist_img = self.train_dataset[idx]

        sample = {
            "image": mnist_img["pixel_values"],
        }

        sample = self.extend_sample(self.sparse_dm, self.rgb_image, sample)
        return sample

    def __len__(self):
        return len(self.train_dataset)
