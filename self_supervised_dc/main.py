import os
import pathlib
import hydra
import omegaconf as oc

from .dataset import DCDataModule


@hydra.main(config_path="configs", config_name="config", version_base="1.1.0")
def main(config: oc.DictConfig) -> None:
    oc.OmegaConf.set_struct(config, True)
    with oc.open_dict(config):
        config.use_pose = "photo" in config.train_mode
        # config.pretrained = not config.no_pretrained
        config.result = os.path.join("..", "results")
        config.use_rgb = ("rgb" in config.input) or config.use_pose
        config.use_d = "d" in config.input
        config.use_g = "g" in config.input
    dm = DCDataModule(config)
    dm.setup()


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(pathlib.Path(__file__).parent.resolve())
    main()
