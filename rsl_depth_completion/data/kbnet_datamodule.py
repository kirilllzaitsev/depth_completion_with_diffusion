from typing import Any, Dict, Optional, Tuple

import torch
from kbnet import data_utils
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

# from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from rsl_depth_completion.data.kbnet import kbnet_dataset
from kbnet.datasets import KBNetTrainingDataset


class KBnetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        ds_config: dict = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def input_img_size(self):
        return (224, 224)

    @property
    def sparse_dm_size(self):
        return (352, 1216)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            ds_config: Any = self.hparams.ds_config
            train_image_paths = data_utils.read_paths(ds_config.train_image_path)
            train_sparse_depth_paths = data_utils.read_paths(
                ds_config.train_sparse_depth_path
            )
            train_intrinsics_paths = data_utils.read_paths(
                ds_config.train_intrinsics_path
            )
            val_image_paths = data_utils.read_paths(ds_config.val_image_path)
            val_intrinsics_paths = data_utils.read_paths(ds_config.test_intrinsics_path)
            val_sparse_depth_paths = data_utils.read_paths(
                ds_config.val_sparse_depth_path
            )
            val_ground_truth_paths = data_utils.read_paths(
                ds_config.val_ground_truth_path
            )

            test_image_paths = data_utils.read_paths(ds_config.test_image_path)
            test_sparse_depth_paths = data_utils.read_paths(
                ds_config.test_sparse_depth_path
            )
            test_intrinsics_paths = data_utils.read_paths(
                ds_config.test_intrinsics_path
            )

            trainset = KBNetTrainingDataset(
                image_paths=train_image_paths,
                sparse_depth_paths=train_sparse_depth_paths,
                intrinsics_paths=train_intrinsics_paths,
                shape=ds_config.shape,
                random_crop_type=ds_config.random_crop_type,
            )

            train_len = int(len(trainset) * 0.9)
            val_len = len(trainset) - train_len
            lengths = [train_len, val_len]
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )

            self.data_test = kbnet_dataset.CustomKBNetInferenceDataset(
                image_paths=val_image_paths,
                sparse_depth_paths=val_sparse_depth_paths,
                intrinsics_paths=val_intrinsics_paths,
                ground_truth_paths=val_ground_truth_paths,
            )
            self.data_predict = kbnet_dataset.CustomKBNetInferenceDataset(
                image_paths=test_image_paths,
                sparse_depth_paths=test_sparse_depth_paths,
                intrinsics_paths=test_intrinsics_paths,
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

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_predict,
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
    ds = KBnetDataModule()
