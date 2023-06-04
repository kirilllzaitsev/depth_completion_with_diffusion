from functools import partial

import torch
import torchvision as tv
from rsl_depth_completion.conditional_diffusion.load_data_kitti import KITTIDMDataset
from rsl_depth_completion.conditional_diffusion.load_data_mnist import (
    MNISTDMDataset,
    mnist_transforms,
)

# from load_data_kitti import KITTIDMDataset


class DMDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transform, *args, **kwargs):
        self.ds = ds
        self.eval_batch = ds.eval_batch
        self.transform = transform

    def __getitem__(self, index):
        sample = self.ds[index]

        # cond_image /= 255.0

        sample["input_img"] = self.transform(sample["input_img"])

        return sample

    def __len__(self):
        return len(self.ds)


from functools import lru_cache


@lru_cache
def load_data(cfg, ds_name="mnist", do_overfit=False, **ds_kwargs):
    mnist_ds_internal_transform = partial(
        mnist_transforms,
        transform=tv.transforms.Compose(
            [
                tv.transforms.Resize(cfg.input_img_size, antialias=True),
                tv.transforms.Lambda(lambda x: x),
            ]
        ),
    )
    if ds_name == "mnist":
        sub_ds = MNISTDMDataset(
            img_transform=mnist_ds_internal_transform,
            cfg=cfg,
            **ds_kwargs,
        )
    elif ds_name == "kitti":
        sub_ds = KITTIDMDataset(
            cfg=cfg,
            **ds_kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    post_transforms = [
        # tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        # tv.transforms.Lambda(lambda t: (t * 2) - 1),
    ]
    if ds_name == "mnist":
        post_transforms = (
            [tv.transforms.Resize(cfg.input_img_size, antialias=True)]
        ) + post_transforms

    post_transform = tv.transforms.Compose(post_transforms)
    ds = DMDataset(sub_ds, transform=post_transform)

    subset_range = range(0, cfg.batch_size) if do_overfit else range(0, 400)

    ds_subset = torch.utils.data.Subset(
        ds,
        subset_range,
    )
    if cfg.do_train_val_split:
        train_size = int(0.8 * len(ds_subset))
        test_size = len(ds_subset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            ds_subset, [train_size, test_size]
        )
    else:
        train_dataset = ds_subset
        valid_dataset = ds_subset

    dl_opts = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "drop_last": True,
        "pin_memory": True,
        "prefetch_factor": 2
    }
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **dl_opts, shuffle=False if do_overfit else True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **dl_opts, shuffle=False
    )
    return ds, train_dataloader, valid_dataloader


if __name__ == "__main__":
    from rsl_depth_completion.conditional_diffusion.config import cfg as cfg_cls

    cfg = cfg_cls()
    ds_kwargs = dict(
        use_rgb_as_text_embed=False,
        use_rgb_as_cond_image=True,
        use_cond_image=True,
        use_text_embed=True,
        do_crop=True,
        include_sdm_and_rgb_in_sample=True,
    )
    ds, train_loader, val_loader = load_data(cfg, ds_name="kitti", **ds_kwargs)
    x = ds[0]
    print(x)
