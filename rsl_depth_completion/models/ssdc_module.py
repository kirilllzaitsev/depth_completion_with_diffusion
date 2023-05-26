import numpy as np

from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from kbnet.net_utils import OutlierRemoval
from rsl_depth_completion.metrics import TotalMetric


class MNISTLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        outlier_removal: OutlierRemoval,
        transforms: dict,
        config: Any,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.outlier_removal = outlier_removal
        self.config = config
        self.train_transforms = transforms["train"]
        self.val_transforms = transforms["val"]

        self.test_total = TotalMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    

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
        self.log(
            "test/mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test/rmse",
            rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test/imae",
            imae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test/irmse",
            irmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_depth = self.model_step(batch, self.val_transforms)
        return output_depth


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
