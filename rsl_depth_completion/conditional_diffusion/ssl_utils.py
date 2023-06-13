import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
from kbnet import log_utils, losses, net_utils, networks
from rsl_depth_completion.models.benchmarking.calibrated_backprojection_network.kbnet import (
    eval_utils,
)


def plot_losses(losses, rec_losses):
    import pandas as pd

    fig, axs = plt.subplots(1, len(losses), figsize=(25, 5))
    fig.suptitle("Main losses")
    for i, (k, v) in enumerate(losses.items()):
        axs[i].plot(v)
        axs[i].set_title(k)
        axs[i].set_xlabel("Step")
        axs[i].set_ylabel("Loss")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.7)
    plt.show()
    pd.DataFrame(rec_losses).plot(title="Reconstruction losses", figsize=(4, 2))

    plt.show()


def calc_error_to_gt(
    output_depth, ground_truth, min_evaluate_depth=1.5, max_evaluate_depth=100
):
    validity_map = torch.where(
        ground_truth > 0, torch.ones_like(ground_truth), ground_truth
    )
    ground_truth = torch.stack([ground_truth, validity_map], axis=-1).cpu().numpy()
    ground_truth = np.squeeze(ground_truth)
    output_depth = np.squeeze(output_depth.detach().cpu().numpy())
    # if len(output_depth.shape) == 2:
    #     output_depth = output_depth[ np.newaxis, :, :]

    validity_map = ground_truth[..., 1]
    ground_truth = ground_truth[..., 0]

    validity_mask = np.where(validity_map > 0, 1, 0)
    min_max_mask = np.logical_and(
        ground_truth > min_evaluate_depth,
        ground_truth < max_evaluate_depth,
    )
    mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

    output_depth = output_depth[mask]
    ground_truth = ground_truth[mask]

    mae = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
    rmse = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
    imae = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
    irmse = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)
    return mae, rmse, imae, irmse


def plot_grads(depth_grads, pose_grads, topk=5, bottomk=5):
    import pandas as pd

    depth_grads = {
        k: v
        for k, v in sorted(
            depth_grads.items(), key=lambda x: np.mean(x[1]), reverse=True
        )
    }
    pose_grads = {
        k: v
        for k, v in sorted(
            pose_grads.items(), key=lambda x: np.mean(x[1]), reverse=True
        )
    }
    if topk is not None:
        depth_grads = {k: v for k, v in list(depth_grads.items())[: topk + 1]}
        pose_grads = {k: v for k, v in list(pose_grads.items())[: topk + 1]}
    elif bottomk is not None:
        depth_grads = {k: v for k, v in list(depth_grads.items())[-bottomk - 1 :]}
        pose_grads = {k: v for k, v in list(pose_grads.items())[-bottomk - 1 :]}

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # fig.suptitle("Depth (left) and pose (right) gradients")
    def display_grads(grads, ax, title):
        print(grads)
        grads = {**grads}
        for k, v in grads.items():
            for i, grad in enumerate(v):
                if np.isnan(grad):
                    v[i] = -1
            # if np.isnan(v).any():
            #     print(f"NaN in {k}. {title} invalid.")
            #     return
        ax.set_xlabel("Step")
        ax.set_ylabel("L1 norm")
        try:
            pd.DataFrame({k: v for k, v in grads.items() if v}).plot(title=title, ax=ax, legend=False)
        except:
            ax.set_title(f"{title} (empty)")

    display_grads(depth_grads, axs[0], "Depth model gradients")
    display_grads(pose_grads, axs[1], "Pose model gradients")
    plt.show()
    return fig


def plot_train_depths_overall(output_depths, save_dir=None, save_subdir_prefix=""):
    figs = []
    batch_size = output_depths[0]
    for idx in range(len(batch_size)):
        pred_ckpts = np.linspace(
            0, len(output_depths) - 1, min(20, len(output_depths))
        ).astype(int)
        n_rows = len(pred_ckpts) // 5 if len(pred_ckpts) % 5 == 0 else len(pred_ckpts) // 5 + 1
        n_cols = min(5, len(pred_ckpts))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        for i, ckpt in enumerate(pred_ckpts):
            pred = output_depths[ckpt][idx].detach().cpu().numpy().squeeze()
            if n_rows == 1:
                if n_cols == 1:
                    ax = axs
                else:
                    ax = axs[i % 5]
            else:
                ax = axs[i // 5, i % 5]
            ax.imshow(pred)
            ax.set_title(f"Epoch {ckpt}")
            ax.axis("off")
            if save_dir is not None:
                dst_dir = os.path.join(save_dir, f"{save_subdir_prefix}/{idx}")
                os.makedirs(dst_dir, exist_ok=True)
                filename = f"epoch_{ckpt}.png"
                cv2.imwrite(os.path.join(dst_dir, filename), pred)
                # plt.savefig(os.path.join(dst_dir, filename))
        plt.tight_layout()
        figs.append(fig)
    return figs


def compute_triplet_loss(
    image0,
    image1,
    image2,
    output_depth0,
    sparse_depth0,
    validity_map_depth0,
    intrinsics,
    pose01,
    pose02,
    w_color=0.15,
    w_structure=0.95,
    w_sparse_depth=0.60,
    w_smoothness=0.04,
):
    """
    Computes loss function
    l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

    Arg(s):
        image0 : torch.Tensor[float32]
            N x 3 x H x W image at time step t
        image1 : torch.Tensor[float32]
            N x 3 x H x W image at time step t-1
        image2 : torch.Tensor[float32]
            N x 3 x H x W image at time step t+1
        output_depth0 : torch.Tensor[float32]
            N x 1 x H x W output depth at time t
        sparse_depth0 : torch.Tensor[float32]
            N x 1 x H x W sparse depth at time t
        validity_map_depth0 : torch.Tensor[float32]
            N x 1 x H x W validity map of sparse depth at time t
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics matrix
        pose01 : torch.Tensor[float32]
            N x 4 x 4 relative pose from image at time t to t-1
        pose02 : torch.Tensor[float32]
            N x 4 x 4 relative pose from image at time t to t+1
        w_color : float
            weight of color consistency term
        w_structure : float
            weight of structure consistency term (SSIM)
        w_sparse_depth : float
            weight of sparse depth consistency term
        w_smoothness : float
            weight of local smoothness term
    Returns:
        torch.Tensor[float32] : loss
        dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
    """

    shape = image0.shape
    validity_map_image0 = torch.ones_like(sparse_depth0)

    # Backproject points to 3D camera coordinates
    points = net_utils.backproject_to_camera(output_depth0, intrinsics, shape)

    # Reproject points onto image 1 and image 2
    target_xy01 = net_utils.project_to_pixel(points, pose01, intrinsics, shape)
    target_xy02 = net_utils.project_to_pixel(points, pose02, intrinsics, shape)

    # Reconstruct image0 from image1 and image2 by reprojection
    image01 = net_utils.grid_sample(image1, target_xy01, shape)
    image02 = net_utils.grid_sample(image2, target_xy02, shape)

    """
    Essential loss terms
    """
    # Color consistency loss function
    loss_color01 = losses.color_consistency_loss_func(
        src=image01, tgt=image0, w=validity_map_image0
    )
    loss_color02 = losses.color_consistency_loss_func(
        src=image02, tgt=image0, w=validity_map_image0
    )
    loss_color = loss_color01 + loss_color02

    # Structural consistency loss function
    loss_structure01 = losses.structural_consistency_loss_func(
        src=image01, tgt=image0, w=validity_map_image0
    )
    loss_structure02 = losses.structural_consistency_loss_func(
        src=image02, tgt=image0, w=validity_map_image0
    )
    loss_structure = loss_structure01 + loss_structure02

    # Sparse depth consistency loss function
    loss_sparse_depth = losses.sparse_depth_consistency_loss_func(
        src=output_depth0, tgt=sparse_depth0, w=validity_map_depth0
    )

    # Local smoothness loss function
    loss_smoothness = losses.smoothness_loss_func(predict=output_depth0, image=image0)

    # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
    loss = (
        w_color * loss_color +
        w_structure * loss_structure
        + w_sparse_depth * loss_sparse_depth
        + w_smoothness * loss_smoothness
    )

    loss_info = {
        "loss_color": loss_color,
        "loss_structure": loss_structure,
        "loss_sparse_depth": loss_sparse_depth,
        "loss_smoothness": loss_smoothness,
        "loss": loss,
        "image01": image01,
        "image02": image02,
    }

    return loss, loss_info
