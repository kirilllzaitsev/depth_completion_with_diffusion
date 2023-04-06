from typing import Any

import torch
from kbnet.kbnet_model import KBNetModel
from kbnet.net_utils import OutlierRemoval
from lightning import LightningModule
from rsl_depth_completion.metrics import TotalMetric
import numpy as np


class KBnetLitModule(LightningModule):
    def __init__(
        self,
        net: KBNetModel,
        outlier_removal: OutlierRemoval,
        transforms: dict,
        config: Any,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.outlier_removal = outlier_removal
        self.config = config
        self.train_transforms = transforms["train"]
        self.val_transforms = transforms["val"]

        self.test_total = TotalMetric()

    def load_from_checkpoint(self, ckpt_path):
        self.net.restore_model(ckpt_path)
        if self.device.type == "cpu":
            self.net.encoder = self.net.encoder.module
            self.net.decoder = self.net.decoder.module
            self.net.sparse_to_dense_pool = self.net.sparse_to_dense_pool.module
        return self

    def forward(self, x: torch.Tensor):
        return torch.randn(x.shape[0], 10)

    def model_step(self, batch: Any, transforms):
        image, sparse_depth, intrinsics = batch

        # Validity map is where sparse depth is available
        validity_map_depth = torch.where(
            sparse_depth > 0, torch.ones_like(sparse_depth), sparse_depth
        )

        # Remove outlier points and update sparse depth and validity map
        (
            filtered_sparse_depth,
            filtered_validity_map_depth,
        ) = self.outlier_removal.remove_outliers(
            sparse_depth=sparse_depth, validity_map=validity_map_depth
        )

        (
            [image],
            [sparse_depth],
            [filtered_sparse_depth, filtered_validity_map_depth],
        ) = transforms.transform(
            images_arr=[image],
            range_maps_arr=[sparse_depth],
            validity_maps_arr=[filtered_sparse_depth, filtered_validity_map_depth],
            random_transform_probability=0.0,
        )

        # Forward through network
        output_depth = self.net.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map_depth=filtered_validity_map_depth,
            intrinsics=intrinsics,
        )
        return output_depth

    def eval_step(self, batch: Any, batch_idx: int):

        ground_truth = batch.pop().cpu().numpy()
        output_depth = self.model_step(batch, self.val_transforms)
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        ground_truth = np.squeeze(ground_truth)

        validity_map = ground_truth[:, :, 1]
        ground_truth = ground_truth[:, :, 0]

        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > self.config.min_evaluate_depth,
            ground_truth < self.config.max_evaluate_depth,
        )
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        return output_depth, ground_truth

    def validation_step(self, batch: Any, batch_idx: int):
        output_depth, ground_truth = self.eval_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int):
        output_depth, ground_truth = self.eval_step(batch, batch_idx)
        mae, rmse, imae, irmse = self.test_total(output_depth, ground_truth)
        self.log("test/mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/rmse",
            rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/imae",
            imae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/irmse",
            irmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_depth = self.model_step(batch, self.val_transforms)
        return output_depth


if __name__ == "__main__":
    _ = KBnetLitModule(None, None, None, {})
