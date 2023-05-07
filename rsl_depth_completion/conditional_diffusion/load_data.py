from functools import partial

import torch
import torchvision as tv
from load_data_mnist import MNISTDMDataset, mnist_transforms
from rsl_depth_completion.conditional_diffusion.config import cfg
from rsl_depth_completion.conditional_diffusion.load_data_kitti import KITTIDMDataset

# from load_data_kitti import KITTIDMDataset


class DMDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transform, *args, **kwargs):
        self.ds = ds
        self.transform = transform

    def __getitem__(self, index):
        sample = self.ds[index]

        # cond_image /= 255.0

        sample["image"] = self.transform(sample["image"])

        return sample

    def __len__(self):
        return len(self.ds)


img_size = (64, 64)

mnist_ds_internal_transform = partial(
    mnist_transforms,
    transform=tv.transforms.Compose(
        [
            tv.transforms.Resize(img_size, antialias=True),
            tv.transforms.Lambda(lambda x: x),
        ]
    ),
)


def load_data(ds_name="mnist", do_debug=False, **ds_kwargs):
    if ds_name == "mnist":
        sub_ds = MNISTDMDataset(
            img_transform=mnist_ds_internal_transform,
            **ds_kwargs,
        )
    elif ds_name == "kitti":
        sub_ds = KITTIDMDataset(
            **ds_kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    post_transform = tv.transforms.Compose(
        [
            # tv.transforms.RandomHorizontalFlip(),
            # tv.transforms.Resize(img_size, antialias=True),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )
    ds = DMDataset(sub_ds, transform=post_transform)

    if do_debug:
        BATCH_SIZE = 1
        NUM_WORKERS = 0
    elif cfg.is_cluster:
        BATCH_SIZE = 4
        NUM_WORKERS = min(20, BATCH_SIZE)
    else:
        BATCH_SIZE = 2
        NUM_WORKERS = 0

    subset_range = range(0, BATCH_SIZE) if do_debug else range(0, 400)

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
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "drop_last": True,
    }
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **dl_opts, shuffle=False if do_debug else True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **dl_opts, shuffle=False
    )
    return ds, train_dataloader, valid_dataloader


if __name__ == "__main__":
    ds_kwargs = dict(
        use_rgb_as_text_embed=False,
        use_rgb_as_cond_image=True,
        use_cond_image=True,
        use_text_embed=True,
        do_crop=True,
        include_sdm_and_rgb_in_sample=True,
    )
    ds, train_loader, val_loader = load_data(ds_name="kitti", **ds_kwargs)
    x = ds[0]
    print(x)
