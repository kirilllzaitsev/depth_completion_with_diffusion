# from lightning.pytorch.trainer import Trainer
import lightning
import lightning.pytorch.trainer
import torch

from rsl_depth_completion.models.kbnet_module import (
    KBnetLitModule,
)


class Trainer(lightning.pytorch.trainer.Trainer):
    @classmethod
    def load_from_checkpoint(cls, ckpt_path):
        if "benchmarking" in ckpt_path:
            return KBnetLitModule.load_from_checkpoint(ckpt_path)
        else:
            return super().load_from_checkpoint(ckpt_path)


if __name__ == "__main__":
    Trainer.load_from_checkpoint(
        "/media/master/wext/msc_studies/second_semester/research_project/project/rsl_depth_completion/data/ckpts/benchmarking/kbnet-kitti.pth"
    )
