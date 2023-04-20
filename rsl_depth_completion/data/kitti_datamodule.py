from typing import Any, Dict, Optional, Tuple

import omegaconf as oc
import torch
from lightning import LightningDataModule
from rsl_depth_completion.data.kitti.kitti_dataset import KittiDCDataset, CustomKittiDCDataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from kbnet import data_utils


class KittiDataModule(LightningDataModule):
    def __init__(
        self,
        ds_config: Any = None,
        data_dir: str = "data/",
        train_val_split_frac: Tuple[float, float] = (0.9, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def input_img_size(self):
        return (352, 1216)

    @property
    def sparse_dm_size(self):
        return (352, 1216)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            ds_config: Any = self.hparams.ds_config
            with oc.open_dict(ds_config):
                ds_config.use_pose = "photo" in ds_config.train_mode
                ds_config.result = ds_config.result_dir
                ds_config.use_rgb = ("rgb" in ds_config.input) or ds_config.use_pose
                ds_config.use_d = "d" in ds_config.input
                ds_config.use_g = "g" in ds_config.input
            trainset = KittiDCDataset(
                ds_config,
                path_to_calib_files=ds_config.path_to_calib_files,
                path_to_depth_completion_dir=ds_config.path_to_depth_completion_dir,
                train=True,
            )

            train_len = int(len(trainset) * self.hparams.train_val_split_frac[0])
            val_len = len(trainset) - train_len
            lengths = [train_len, val_len]
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )

            testset = KittiDCDataset(
                ds_config,
                path_to_calib_files=ds_config.path_to_calib_files,
                path_to_depth_completion_dir=ds_config.path_to_depth_completion_dir,
                train=False,
            )
            dataset = ConcatDataset(datasets=[trainset, testset])

            test_image_paths = data_utils.read_paths(ds_config.test_image_path)
            test_sparse_depth_paths = data_utils.read_paths(
                ds_config.test_sparse_depth_path
            )
            test_intrinsics_paths = data_utils.read_paths(
                ds_config.test_intrinsics_path
            )

            # trainset = KBNetTrainingDataset(
            #     image_paths=train_image_paths,
            #     sparse_depth_paths=train_sparse_depth_paths,
            #     intrinsics_paths=train_intrinsics_paths,
            #     shape=ds_config.shape,
            #     random_crop_type=ds_config.random_crop_type,
            # )

            # train_len = int(len(trainset) * 0.9)
            # val_len = len(trainset) - train_len
            # lengths = [train_len, val_len]
            # self.data_train, self.data_val = random_split(
            #     dataset=trainset,
            #     lengths=lengths,
            #     generator=torch.Generator().manual_seed(42),
            # )

            # self.data_test = CustomKittiDCDataset(
            #     image_paths=val_image_paths,
            #     sparse_depth_paths=val_sparse_depth_paths,
            #     intrinsics_paths=val_intrinsics_paths,
            #     ground_truth_paths=val_ground_truth_paths,
            # )
            self.data_predict = CustomKittiDCDataset(
                ds_config=ds_config,
                image_paths=test_image_paths,
                sparse_depth_paths=test_sparse_depth_paths,
                intrinsics_paths=test_intrinsics_paths,
            )

    def get_dataloader(self, dataset, shuffle: bool = True):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.data_train)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test, shuffle=False)

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
