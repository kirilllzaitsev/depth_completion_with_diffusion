import os

import dotenv
import torch
import yaml

dotenv.load_dotenv()


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
        if path is not None:
            self.load_from_file(path)

        self.num_workers = min(os.cpu_count(), max(self.batch_size, self.num_gpus))

    do_sample = True
    do_overfit = None
    do_train_val_split = False
    do_lr_schedule = True
    do_early_stopping = True
    other_tags = []
    exp_targets = []

    # disabled = not is_cluster
    # disabled = True
    disabled = False
    input_img_sdm_interpolation_mode = "infill"
    cond_img_sdm_interpolation_mode = "interpolate"

    use_triplet_loss = True

    lr_schedule_cfg = lr_schedule_cfg
    early_stop_cfg = early_stop_cfg

    dim = 32
    input_channels = 1
    timesteps = 200
    cond_scale = 8.0
    input_res = 256
    input_img_size = (input_res, input_res)
    memory_efficient = False  # results in a one-time memory burst that causes OOM
    num_resnet_blocks = 2
    auto_normalize_img = True

    use_validity_map_depth = False
    sz_loss_weight = 0.4

    use_super_res = False
    super_res_img_size = (input_res, input_res)

    unets_output_res = [256]
    if use_super_res:
        unets_output_res.append(super_res_img_size[0])
    stop_at_unet_number = 2

    fp16 = True
    max_batch_size = num_gpus

    num_epochs = None
    train_one_epoch = False
    lr = None
    sampling_freq = None
    do_save_inputs_every_batch = False

    # eval_batch_path = "eval_batch.pt"
    eval_batch_path = "eval_batch_ssl.pt"

    max_depth = 80

    seed = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tmpdir = os.getenv("TMPDIR")
    cluster_logdir = "/cluster/scratch/kzaitse/logs"
    path_to_project_dir = os.environ["path_to_project_dir"]

    def params(self):
        attrs_to_exclude = [
            "path_to_project_dir",
            "base_kitti_dataset_dir",
        ]
        return {
            **{
                k: v
                for k, v in vars(self).items()
                if not k.startswith("__")
                and not callable(v)
                and k not in attrs_to_exclude
            },
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
    cfg.base_kitti_dataset_dir = os.path.join(
        cfg.tmpdir, os.environ["base_kitti_dataset_dir"]
    )
else:
    cfg.base_kitti_dataset_dir = os.environ["base_kitti_dataset_dir"]

if cfg.use_super_res:
    cfg.other_tags.append(f"super_res_{cfg.super_res_img_size}")


if __name__ == "__main__":
    print(cfg.params())
