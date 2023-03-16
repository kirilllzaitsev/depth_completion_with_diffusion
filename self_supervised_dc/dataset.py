import os
import pathlib

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict


class DCDataModule(pl.LightningDataModule):
    def __init__(self, ds_config):
        super().__init__()
        self.config = ds_config
        self.batch_size = ds_config.batch_size
        self.num_workers = ds_config.num_workers
        self.split = ds_config.split
        self.subsplit = (
            ds_config.subsplit if ds_config.subsplit is not None else "train"
        )

    def setup(self, stage=None):
        self.dataset = hydra.utils.instantiate(self.config.dataset, self.config)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


@hydra.main(config_path="configs", config_name="config", version_base="1.1.0")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.use_pose = "photo" in config.train_mode
        # config.pretrained = not config.no_pretrained
        config.result = os.path.join("..", "results")
        config.use_rgb = ("rgb" in config.input) or config.use_pose
        config.use_d = "d" in config.input
        config.use_g = "g" in config.input
    dm = DCDataModule(config)
    dm.setup()
    print(len(dm.dataset))


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(pathlib.Path(__file__).parent.resolve())
    main()
