from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class UniformMNIST(Dataset):
    def __init__(self, transform=None,train=True,*args,**kwargs):
        self.transform = transform
        num_samples = 60_000 if train else 10_000
        self.data = torch.zeros((num_samples, 28, 28), dtype=torch.float32)
        idx_limits_per_class = torch.linspace(num_samples // 10, num_samples, 10).long()
        self.targets = torch.zeros(num_samples, dtype=torch.long)
        for i, idx_limit_per_class in enumerate(idx_limits_per_class):
            color = torch.rand(1)
            if i + 1 == len(idx_limits_per_class):
                continue
            if i == 0:
                self.data[:idx_limit_per_class] = color
                self.targets[:idx_limit_per_class] = i
            else:
                self.data[idx_limit_per_class : idx_limits_per_class[i + 1]] = color
                self.targets[idx_limit_per_class : idx_limits_per_class[i + 1]] = i
        self.data = self.data.unsqueeze(1)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class MNISTDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    # def prepare_data(self):
    #     """Download data if needed.

    #     Do not use it to assign state (self.x = y).
    #     """
    #     MNIST(root_dir=self.hparams.data_dir, train=True, download=True)
    #     MNIST(root_dir=self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = UniformMNIST(
                root_dir=self.hparams.data_dir, train=True, transform=self.transforms
            )
            testset = UniformMNIST(
                root_dir=self.hparams.data_dir, train=False, transform=self.transforms
            )
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = MNISTDataModule()
