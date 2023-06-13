import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ip_basic import depth_map_utils
from rsl_depth_completion.conditional_diffusion.ssl_utils import calc_error_to_gt
from scipy.interpolate import griddata


def load_extractors():
    from transformers import CLIPModel, CLIPProcessor

    extractor_model_ref = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(extractor_model_ref)
    processor = CLIPProcessor.from_pretrained(extractor_model_ref)
    return model, processor


def get_model_params(parameters_dir):
    params = {}
    for file in os.listdir(f"{parameters_dir}"):
        if file.endswith(".json"):
            with open(f"{parameters_dir}/{file}") as f:
                params[os.path.splitext(file)[0]] = json.load(f)

    return params


def infill_sparse_depth(sdm, method="nearest"):
    # Find the indices of the missing values
    missing_indices = np.argwhere(sdm == 0)

    # Find the indices of the valid values
    valid_indices = np.argwhere(~(sdm == 0))

    # Extract the valid depth values and their corresponding indices
    valid_depth = sdm[~(sdm == 0)]
    valid_coords = valid_indices.astype(float)

    # Perform nearest neighbor interpolation to estimate the missing depth values
    interp_depth = griddata(valid_coords, valid_depth, missing_indices, method=method)

    # Replace the missing values in the original depth map with the interpolated values
    near_img = sdm.copy()
    near_img[sdm == 0] = interp_depth
    return near_img


def interpolate_sparse_depth(sdm, do_multiscale=False, *args, **kwargs):
    """See depth_map_utils.fill_in_fast"""
    if do_multiscale:
        ddm, _ = depth_map_utils.fill_in_multiscale(
            sdm.astype("float32"), *args, **kwargs
        )
    else:
        ddm = depth_map_utils.fill_in_fast(sdm.astype("float32"), *args, **kwargs)
    return ddm


def overlay_sparse_depth_and_rgb(rgb, sdm):
    sdm = sdm.squeeze().numpy()
    rgb = rgb.squeeze().numpy().transpose(1, 2, 0)
    sdm = interpolate_sparse_depth(sdm, max_depth=80)
    sdm = cv2.cvtColor(sdm, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(rgb, 0.5, sdm, 0.5, 0)


def plot_sample(x, items):
    fix, ax = plt.subplots(len(items) // 2, 2, figsize=(15, 10))
    for idx, item in enumerate(items):
        img = x[item].numpy().transpose(1, 2, 0)

        ax_idx = idx // 2, idx % 2
        ax[ax_idx].imshow(img, cmap="gray")
        ax[ax_idx].set_title(
            item,
        )
    plt.show()


def plot_source_and_reconstruction(src, rec):
    def cast_to_tensor(x):
        if not isinstance(type(x), torch.Tensor):
            if not isinstance(type(x), np.ndarray):
                x = np.array(x)
            x = torch.tensor(x)
        return x

    src = cast_to_tensor(src)
    rec = cast_to_tensor(rec)
    concatenated_img_and_denoised = torch.concatenate([src, rec], dim=2)
    plt.imshow(concatenated_img_and_denoised.permute(1, 2, 0).cpu().detach().numpy())


def read_paths(filepath, path_to_data_dir=None):
    """
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    """

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip("\n")
            # If there was nothing to read
            if path == "":
                break
            if path_to_data_dir is not None:
                path = path_to_data_dir + "/" + path
            path_list.append(path)

    return path_list


def log_params_to_exp(experiment, params: dict, prefix: str):
    experiment.log_parameters({f"{prefix}/{k}": v for k, v in params.items()})


def dict2mdtable(d, key="Name", val="Value"):
    rows = [f"| {key} | {val} |"]
    rows += ["|--|--|"]
    rows += [f"| {k} | {v} |" for k, v in d.items()]
    return "  \n".join(rows)


def optional_normalize_img(x, scaler=255.0):
    if np.max(x) > 1:
        x = x / scaler
    return x


def rescale_img_to_zero_one_range(x):
    return x / np.max(x)


def log_batch(
    batch,
    step,
    batch_size,
    experiment,
    prefix=None,
    max_depth=80.0,
):
    for k, v in batch.items():
        # if k in ["text_embed", "adj_imgs"]:
        if k in ["gt", "g", "r_mats", "t_vecs", "intrinsics", "text_embed"]:
            continue
        if k == "adj_imgs":
            for idx, img in enumerate(v):
                log_image_comet(
                    step,
                    batch_size,
                    experiment,
                    prefix,
                    f"{k}_{idx}",
                    img,
                )
        else:
            log_image_comet(step, batch_size, experiment, prefix, k, v)


def log_image_comet(step, batch_size, experiment, prefix, k, v):
    if len(v.shape) == 3:
        v = v.unsqueeze(0)
    v = v.cpu().numpy().transpose(0, 2, 3, 1)
    v = rescale_img_to_zero_one_range(v)
    for idx in range(batch_size):
        name = f"{k}_{idx}"
        if prefix is not None:
            name = f"{prefix}/{name}"
        experiment.log_image(
            v[idx],
            name,
            step=step,
        )


def print_metrics(mae, rmse, imae, irmse, comment="", save_csv=False, save_csv_dir=None):
    mae_mean = np.mean(mae)
    rmse_mean = np.mean(rmse)
    imae_mean = np.mean(imae)
    irmse_mean = np.mean(irmse)

    mae_std = np.std(mae)
    rmse_std = np.std(rmse)
    imae_std = np.std(imae)
    irmse_std = np.std(irmse)

    # Print evaluation results to console and file
    print(f"Evaluation results ({comment})")
    print("{:>8}  {:>8}  {:>8}  {:>8}".format("MAE", "RMSE", "iMAE", "iRMSE"))
    print(
        "{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}".format(
            mae_mean, rmse_mean, imae_mean, irmse_mean
        )
    )

    print("{:>8}  {:>8}  {:>8}  {:>8}".format("+/-", "+/-", "+/-", "+/-"))
    print(
        "{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}".format(
            mae_std, rmse_std, imae_std, irmse_std
        )
    )
    if save_csv:
        df = pd.DataFrame(
            {
                "MAE": mae,
                "RMSE": rmse,
                "iMAE": imae,
                "iRMSE": irmse,
            }
        )
        save_path = f"{comment}.csv"
        if save_csv_dir is not None:
            save_path = os.path.join(save_csv_dir, save_path)
        df = df.round(2)
        df.to_csv(save_path)
        print(f"Saved results to {save_path}")


def plot_full_prediction(output_depths, eval_batch, kbnet_predictor, idx_to_use=None, save_csv_dir=None):
    idxs = [idx_to_use] if idx_to_use is not None else range(len(eval_batch["rgb"]))
    kbnet_pred = (
        kbnet_predictor.predict(
            image=eval_batch["rgb"].cuda(),
            sparse_depth=eval_batch["sdm"].cuda(),
            intrinsics=eval_batch["intrinsics"].cuda(),
        )
        .cpu()
        .squeeze(1)
        .detach()
        .numpy()
    )
    figs = []
    maes_kbnet, rmses_kbnet, imaes_kbnet, irmses_kbnet = [], [], [], []
    maes_trainer, rmses_trainer, imaes_trainer, irmses_trainer = [], [], [], []
    for idx in idxs:
        img, pred, sdm, gt, input_img, cond_img, lowres_img = (
            eval_batch["rgb"][idx],
            output_depths[idx],
            eval_batch["sdm"][idx],
            eval_batch["gt"][idx],
            eval_batch["input_img"][idx],
            eval_batch["cond_img"][idx],
            eval_batch["lowres_img"][idx],
        )
        img = img.permute(1, 2, 0).cpu().numpy()
        pred = pred.squeeze().cpu().detach().numpy()
        sdm = sdm.squeeze().cpu().detach().numpy()
        sdm = np.where(sdm > 0, 1, 0)
        gt = gt.squeeze().cpu().detach().numpy()
        input_img = input_img.permute(1, 2, 0).cpu().numpy()
        cond_img = cond_img.permute(1, 2, 0).cpu().numpy()
        lowres_img = lowres_img.permute(1, 2, 0).cpu().numpy()
        fig, axs = plt.subplots(1, 8, figsize=(15, 5))
        axs[0].imshow(img)
        axs[1].imshow(sdm)
        axs[2].imshow(gt.squeeze())
        axs[3].imshow(pred)
        axs[4].imshow(input_img)
        axs[5].imshow(cond_img)
        axs[6].imshow(lowres_img)
        axs[7].imshow(kbnet_pred[idx])
        axs[0].set_title("RGB")
        axs[1].set_title("sparse_depth")
        axs[2].set_title("gt")
        axs[3].set_title("predicted_depth")
        axs[4].set_title("input_image")
        axs[5].set_title("cond_img")
        axs[6].set_title("lowres_img")
        axs[7].set_title("kbnet_pred")

        if save_csv_dir is not None:
            os.makedirs(f"{save_csv_dir}/preds", exist_ok=True)
            cv2.imwrite(f"{save_csv_dir}/preds/kbnet_pred_{idx}.png", kbnet_pred[idx])
            cv2.imwrite(f"{save_csv_dir}/preds/lowres_img_{idx}.png", lowres_img*100)
            cv2.imwrite(f"{save_csv_dir}/preds/cond_img_{idx}.png", cond_img*100)
            cv2.imwrite(f"{save_csv_dir}/preds/input_img_{idx}.png", input_img*100)
            cv2.imwrite(f"{save_csv_dir}/preds/pred_{idx}.png", pred)
            cv2.imwrite(f"{save_csv_dir}/preds/gt_{idx}.png", gt)
            cv2.imwrite(f"{save_csv_dir}/preds/sdm_{idx}.png", sdm*100)
            cv2.imwrite(f"{save_csv_dir}/preds/img_{idx}.png", img*255)
            print(f"Saved rgb for prediction to {save_csv_dir}/preds/img_{idx}.png")
        for ax in axs:
            ax.axis("off")
        plt.show()
        figs.append(fig)

        mae_kbnet, rmse_kbnet, imae_kbnet, irmse_kbnet = calc_error_to_gt(
            torch.tensor(kbnet_pred).cuda()[idx], eval_batch["gt"][idx]
        )
        mae_trainer, rmse_trainer, imae_trainer, irmse_trainer = calc_error_to_gt(
            output_depths.cuda()[idx], eval_batch["gt"][idx]
        )
        maes_kbnet.append(mae_kbnet)
        rmses_kbnet.append(rmse_kbnet)
        imaes_kbnet.append(imae_kbnet)
        irmses_kbnet.append(irmse_kbnet)
        maes_trainer.append(mae_trainer)
        rmses_trainer.append(rmse_trainer)
        imaes_trainer.append(imae_trainer)
        irmses_trainer.append(irmse_trainer)
    print_metrics(
        maes_trainer,
        rmses_trainer,
        imaes_trainer,
        irmses_trainer,
        comment="trainer",
        save_csv=True,
        save_csv_dir=save_csv_dir
    )
    print_metrics(
        maes_kbnet,
        rmses_kbnet,
        imaes_kbnet,
        irmses_kbnet,
        comment="kbnet",
        save_csv=True,
        save_csv_dir=save_csv_dir
    )

    return figs
