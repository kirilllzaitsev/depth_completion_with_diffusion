from typing import Any

import numpy as np
import torch
from kbnet.kbnet_model import KBNetModel
from kbnet.net_utils import OutlierRemoval
from lightning import LightningModule
from rsl_depth_completion.metrics import TotalMetric


class KBnetLitModule(LightningModule):
    def __init__(
        self,
        depth_net: KBNetModel,
        pose_net: torch.nn.Module,
        outlier_removal: OutlierRemoval,
        transforms: dict,
        config: Any,
        ckpt_path: str = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=True)

        self.depth_net = depth_net
        self.pose_net = pose_net
        self.outlier_removal = outlier_removal
        self.config = config
        self.train_transforms = transforms["train"]
        self.val_transforms = transforms["val"]

        self.train_total = TotalMetric()
        self.val_total = TotalMetric()
        self.test_total = TotalMetric()
        self.to(self.device)

    def load_from_checkpoint(self, ckpt_path):
        self.depth_net.restore_model(ckpt_path)
        if self.device.type == "cpu":
            self.depth_net.encoder = self.depth_net.encoder.module
            self.depth_net.decoder = self.depth_net.decoder.module
            self.depth_net.sparse_to_dense_pool = (
                self.depth_net.sparse_to_dense_pool.module
            )
        return self
    
    def to(self, device):
        self.depth_net.to(device)
        self.pose_net.to(device)

    def forward(self, x: torch.Tensor):
        return torch.randn(x.shape[0], 10)

    def eval_model_step(self, batch: Any, transforms):
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
        output_depth = self.depth_net.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map_depth=filtered_validity_map_depth,
            intrinsics=intrinsics,
        )
        return output_depth

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx):
        return [x.to(device) for x in batch]


    def training_step(self, batch: Any, batch_idx: int):
        image0, image1, image2, sparse_depth0, intrinsics = batch

        # Validity map is where sparse depth is available
        validity_map_depth0 = torch.where(
            sparse_depth0 > 0, torch.ones_like(sparse_depth0), sparse_depth0
        )

        # Remove outlier points and update sparse depth and validity map
        (
            filtered_sparse_depth0,
            filtered_validity_map_depth0,
        ) = self.outlier_removal.remove_outliers(
            sparse_depth=sparse_depth0, validity_map=validity_map_depth0
        )

        # Do data augmentation
        (
            [image0, image1, image2],
            [sparse_depth0],
            [filtered_sparse_depth0, filtered_validity_map_depth0],
        ) = self.train_transforms.transform(
            images_arr=[image0, image1, image2],
            range_maps_arr=[sparse_depth0],
            validity_maps_arr=[filtered_sparse_depth0, filtered_validity_map_depth0],
            random_transform_probability=self.config.augmentation_probability,
        )

        # Forward through the network
        output_depth0 = self.depth_net.forward(
            image=image0,
            sparse_depth=sparse_depth0,
            validity_map_depth=filtered_validity_map_depth0,
            intrinsics=intrinsics,
        )

        pose01 = self.pose_net.forward(image0, image1)
        pose02 = self.pose_net.forward(image0, image2)

        # Compute loss function
        loss, loss_info = self.depth_net.compute_loss(
            image0=image0,
            image1=image1,
            image2=image2,
            output_depth0=output_depth0,
            sparse_depth0=filtered_sparse_depth0,
            validity_map_depth0=filtered_validity_map_depth0,
            intrinsics=intrinsics,
            pose01=pose01,
            pose02=pose02,
            w_color=self.config.w_color,
            w_structure=self.config.w_structure,
            w_sparse_depth=self.config.w_sparse_depth,
            w_smoothness=self.config.w_smoothness,
        )

        stage = "train"
        for k, v in loss_info.items():
            if "loss" in k:
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

    def validation_step(self, batch: Any, batch_idx: int):
        output_depth, ground_truth = self.eval_step(batch, batch_idx)

        mae, rmse, imae, irmse = self.calc_metrics(
            self.train_total, output_depth, ground_truth
        )
        self.log_metrics("val", mae, rmse, imae, irmse)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {
                    "params": self.depth_net.parameters(),
                    "weight_decay": self.config.weight_decay_depth,
                },
                {
                    "params": self.pose_net.parameters(),
                    "weight_decay": self.config.weight_decay_pose,
                },
            ],
            lr=self.config.learning_rate,
        )

    def eval_step(self, batch: Any, batch_idx: int):
        ground_truth = batch.pop().cpu().numpy()
        output_depth = self.eval_model_step(batch, self.val_transforms)
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

    def test_step(self, batch: Any, batch_idx: int):
        output_depth, ground_truth = self.eval_step(batch, batch_idx)
        mae, rmse, imae, irmse = self.calc_metrics(
            self.test_total, output_depth, ground_truth
        )
        self.log_metrics("test", mae, rmse, imae, irmse)

    def log_metrics(self, stage, mae, rmse, imae, irmse):
        self.log(
            f"{stage}/mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}/rmse",
            rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}/imae",
            imae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}/irmse",
            irmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def calc_metrics(self, metric_fn, output_depth, ground_truth):
        return metric_fn(output_depth, ground_truth)

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_depth = self.eval_model_step(batch, self.val_transforms)
        return output_depth


if __name__ == "__main__":
    _ = KBnetLitModule(None, None, None, {})
