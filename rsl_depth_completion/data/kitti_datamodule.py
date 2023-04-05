import os
from typing import Any, Dict, Optional, Tuple

import torch
from rsl_depth_completion.data.kitti.kitti_dataset import KittiDCDataset
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

# from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import omegaconf as oc


class KittiDataModule(LightningDataModule):
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
        ds_config: dict = None,
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

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            ds_config = self.hparams.ds_config
            # OmegaConf.set_struct(ds_config, True)
            with oc.open_dict(ds_config):
                ds_config.use_pose = "photo" in ds_config.train_mode
                # ds_config.pretrained = not ds_config.no_pretrained
                ds_config.result = "/media/master/wext/msc_studies/second_semester/research_project/project/rsl_depth_completion/data/results"
                # ds_config.result = os.path.join(self.hparams.paths.data_dir, "results")
                ds_config.use_rgb = ("rgb" in ds_config.input) or ds_config.use_pose
                ds_config.use_d = "d" in ds_config.input
                ds_config.use_g = "g" in ds_config.input
            trainset = KittiDCDataset(
                ds_config,
                path_to_calib_files=ds_config.path_to_calib_files,
                path_to_depth_completion_dir=ds_config.path_to_depth_completion_dir,
                train=True,
                transform=self.transforms,
            )
            testset = KittiDCDataset(
                ds_config,
                path_to_calib_files=ds_config.path_to_calib_files,
                path_to_depth_completion_dir=ds_config.path_to_depth_completion_dir,
                train=False,
                transform=self.transforms,
            )
            dataset = ConcatDataset(datasets=[trainset, testset])
            lengths = [int(len(dataset)*0.8), int(len(dataset)*0.1)]
            lengths += [len(dataset) - sum(lengths)]
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                # lengths=self.hparams.train_val_test_split,
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
    ds = KittiDataModule()
