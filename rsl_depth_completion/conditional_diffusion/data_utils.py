import os

import cv2
import numpy as np
import requests
import torch
from PIL import Image
import json



def update_eval_batch_file(
    eval_batch,
    cfg,
    image_paths,
    do_resave_eval_batch=True,
    filename="eval_batch_rand_rgb_and_sdm.pt",
):
    if do_resave_eval_batch:
        meta_filename = filename.replace(".pt", "_meta.json")
        for k in ["text_embed", "cond_img"]:
            if k in eval_batch:
                del eval_batch[k]

        update_dict = {
            **eval_batch,
        }
        meta = json.load(open(meta_filename)) if os.path.exists(meta_filename) else {}
        meta[cfg.input_res] = {
            "image_paths": image_paths,
        }
        if os.path.exists(filename):
            eval_batch_total = torch.load(filename)
            eval_batch_total[cfg.input_res] = update_dict
        else:
            eval_batch_total = {cfg.input_res: update_dict}
        torch.save(eval_batch_total, filename)
        json.dump(meta, open(meta_filename, "w"))
        print(f"saved for {cfg.input_res}")


def fill_eval_batch_with_coco(eval_batch, cfg, do_fill_in_eval_batch, max_depth=80):
    """
    # do_fill_in_eval_batch = False
    # fill_eval_batch_with_coco(eval_batch,cfg,do_fill_in_eval_batch, max_depth=cfg.max_depth)
    """
    if do_fill_in_eval_batch:
        for i, url in enumerate(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://farm8.staticflickr.com/7138/8152187319_2bbe6765f5_z.jpg",
            ]
        ):
            image = np.array(
                Image.open(requests.get(url, stream=True).raw).resize(
                    (cfg.input_res, cfg.input_res)
                )
            )
            rgb = torch.from_numpy(image)
            eval_batch["rgb"][i] = rgb.permute(2, 0, 1) / 255
            eval_batch["sdm"][i] = (
                torch.from_numpy(
                    cv2.cvtColor((image), cv2.COLOR_RGB2GRAY).astype(np.float32)
                ).unsqueeze(0)
                / max_depth
            )
