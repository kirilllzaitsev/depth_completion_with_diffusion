import os

import torch
import yaml


class lr_schedule_cfg:
    patience = 2
    min_lr = 1e-8
    factor = 0.5

    @classmethod
    def params(cls):
        return {
            "patience": cls.patience,
            "min_lr": cls.min_lr,
            "factor": cls.factor,
        }


class early_stop_cfg:
    patience = 4
    min_delta = 1e-3

    @classmethod
    def params(cls):
        return {
            "patience": cls.patience,
            "min_delta": cls.min_delta,
        }


class cfg:
    # ds_name = "mnist"
    ds_name = "kitti"
    default_file = "configs/overfit.yaml"
    # default_file = "configs/full_dataset.yaml"

    batch_size = None
    num_workers = None
    do_save_model = None
    do_save_last_model = True
    is_cluster = os.path.exists("/cluster")

    def load_from_file(self, path):
        params = yaml.safe_load(open(path))
        for k, v in params.items():
            try:
                v = eval(v)
            except:
                pass
            setattr(self, k, v)

    num_gpus = torch.cuda.device_count()

    def __init__(self, path=None):
        env_specific_config_file = (
            "configs/cluster.yaml" if self.is_cluster else "configs/local.yaml"
        )
        if os.path.exists(env_specific_config_file):
            self.load_from_file(env_specific_config_file)
        if path is not None:
            self.load_from_file(path)

        if self.is_cluster:
            self.batch_size = 2 if self.ds_name == "mnist" else 2
            self.num_workers = min(os.cpu_count(), max(self.batch_size, self.num_gpus))
        else:
            self.batch_size = 1
            self.num_workers = 0
        self.other_tags.append(f"bs_{self.batch_size}")

    do_sample = True
    do_overfit = None
    do_train_val_split = False
    do_lr_schedule = True
    do_early_stopping = True
    # other_tags = ["sdm_interpolation_mode"]
    other_tags = []

    disabled = not is_cluster
    # disabled = True
    # sdm_interpolation_mode = "interpolate"
    sdm_interpolation_mode = "infill"

    lr_schedule_cfg = lr_schedule_cfg
    early_stop_cfg = early_stop_cfg

    dim = 32
    input_channels = 1
    timesteps = 200
    cond_scale = 8.0
    use_super_res = True
    super_res_img_size = (256, 256)
    input_img_size = (64, 64)
    memory_efficient = False
    num_resnet_blocks = 1

    fp16 = True
    max_batch_size = num_gpus

    num_epochs = None
    train_one_epoch = False
    lr = None
    sampling_freq = None
    do_save_inputs_every_batch = False

    seed = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tmpdir = os.getenv("TMPDIR")
    cluster_logdir = "/cluster/scratch/kzaitse/logs"

    path_to_project_dir = None
    base_kitti_dataset_dir = None

    def params(self):
        return {
            "do_sample": self.do_sample,
            "do_overfit": self.do_overfit,
            "do_train_val_split": self.do_train_val_split,
            "do_lr_schedule": self.do_lr_schedule,
            "do_early_stopping": self.do_early_stopping,
            "dim": self.dim,
            "input_channels": self.input_channels,
            "timesteps": self.timesteps,
            "cond_scale": self.cond_scale,
            "num_epochs": self.num_epochs,
            "train_one_epoch": self.train_one_epoch,
            "use_super_res": self.use_super_res,
            "lr": self.lr,
            "fp16": self.fp16,
            "seed": self.seed,
            "sdm_interpolation_mode": self.sdm_interpolation_mode,
            "memory_efficient": self.memory_efficient,
            "num_resnet_blocks": self.num_resnet_blocks,
            **{
                f"lr_schedule_cfg/{k}": v
                for k, v in self.lr_schedule_cfg.params().items()
            },
            **{
                f"early_stop_cfg/{k}": v
                for k, v in self.early_stop_cfg.params().items()
            },
        }


if cfg.is_cluster:
    cfg.path_to_project_dir = "/cluster/home/kzaitse"
    cfg.base_kitti_dataset_dir = os.path.join(
        cfg.tmpdir, "/cluster/project/rsl/kzaitsev/kitti_dataset"
    )
else:
    cfg.path_to_project_dir = (
        "/media/master/wext/msc_studies/second_semester/research_project/project"
    )
    cfg.base_kitti_dataset_dir = "/media/master/wext/cv_data/kitti-full"

if cfg.use_super_res:
    cfg.other_tags.append(f"super_res_{cfg.super_res_img_size}")


if __name__ == "__main__":
    print(cfg.params())
