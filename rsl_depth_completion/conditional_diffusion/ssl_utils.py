import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_depth_completion.models.benchmarking.calibrated_backprojection_network.kbnet import (
    eval_utils,
)


def plot_losses(losses, rec_losses):
    import pandas as pd

    fig, axs = plt.subplots(1, len(losses), figsize=(20, 5))
    fig.suptitle("Main losses")
    for i, (k, v) in enumerate(losses.items()):
        axs[i].plot(v)
        axs[i].set_title(k)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
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
        try:
            pd.DataFrame({k: v for k, v in grads.items() if v}).plot(title=title, ax=ax)
        except:
            ax.set_title(f"{title} (empty)")

    display_grads(depth_grads, axs[0], "Depth gradients")
    display_grads(pose_grads, axs[1], "Pose gradients")
    plt.show()
