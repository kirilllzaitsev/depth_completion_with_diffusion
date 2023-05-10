import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from imagen_pytorch import Imagen
from rsl_depth_completion.conditional_diffusion.config import cfg
from torchvision.utils import save_image
from tqdm import tqdm

progress_bar = tqdm(total=cfg.num_epochs, disable=False)


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0.0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


def train(
    imagen: Imagen, optimizer, train_dataloader, train_writer, out_dir, do_debug=False
):
    lr_scheduler = LRScheduler(
        optimizer,
        patience=cfg.lr_schedule_cfg.patience,
        min_lr=cfg.lr_schedule_cfg.min_lr,
        factor=cfg.lr_schedule_cfg.factor,
    )
    early_stopping = EarlyStopping(
        patience=cfg.early_stop_cfg.patience, min_delta=cfg.early_stop_cfg.min_delta
    )

    if not cfg.do_lr_schedule:
        lr_scheduler.patience = torch.inf
    if not cfg.do_early_stopping:
        early_stopping.patience = torch.inf

    max_outputs = train_dataloader.batch_size

    eval_batch = next(iter(train_dataloader))
    with train_writer.as_default():
        log_batch_input(eval_batch, epoch=1, max_outputs=max_outputs, prefix="eval")

    for epoch in range(cfg.num_epochs):
        progress_bar.set_description(f"Epoch {epoch}")
        optimizer.zero_grad()
        running_loss = {"loss": 0, "diff_to_orig_img": 0}
        imagen.train()
        if do_debug:
            data_gen = enumerate(train_dataloader)
        else:
            data_gen = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in data_gen:
            images = batch["image"].to(cfg.device)
            if "text_embed" in batch:
                text_embeds = batch["text_embed"].to(cfg.device)
            else:
                text_embeds = None
            if "cond_image" in batch:
                cond_images = batch["cond_image"].to(cfg.device)
            else:
                cond_images = None
            with torch.autocast(cfg.device.type):
                for i in range(1, len(imagen.unets)):
                    losses = imagen(
                        images=images,
                        text_embeds=text_embeds,
                        cond_images=cond_images,
                        unet_number=i,
                    )
                    loss = losses[imagen.pred_objectives[i - 1]]
                    diff_to_orig_img = losses["diff_to_orig_img"]
                    loss.backward()
                    running_loss["loss"] += loss.item()
                    running_loss["diff_to_orig_img"] += diff_to_orig_img.item()
            optimizer.step()

            with train_writer.as_default():
                tf.summary.scalar(
                    "batch/loss",
                    loss.item(),
                    step=epoch * len(train_dataloader) + batch_idx,
                )
                if do_debug and cfg.do_save_inputs_every_batch:
                    log_batch_input(batch, epoch, len(images), prefix="train")
            if do_debug and batch_idx == 0:
                break

        with train_writer.as_default():
            tf.summary.scalar("epoch/loss", running_loss["loss"], step=epoch)
            tf.summary.scalar(
                "epoch/diff_to_orig_img", running_loss["diff_to_orig_img"], step=epoch
            )

        progress_bar.update(1)

        if (epoch - 1) % cfg.sampling_freq == 0 or epoch == cfg.num_epochs - 1:
            # if True:
            print(f"Epoch: {epoch}\t{running_loss}")
            lr_scheduler(running_loss["loss"])
            if early_stopping.early_stop:
                break

            progress_bar.set_postfix(**running_loss)

            if cfg.do_sample:
                imagen.eval()
                with torch.no_grad():
                    samples = imagen.sample(
                        text_embeds=eval_batch["text_embed"].to(cfg.device)
                        if "text_embed" in eval_batch
                        else None,
                        cond_images=eval_batch["cond_image"].to(cfg.device)
                        if "cond_image" in eval_batch
                        else None,
                        cond_scale=cfg.cond_scale,
                        batch_size=train_dataloader.batch_size,
                        stop_at_unet_number=None,
                        return_all_unet_outputs=True,
                    )

                with train_writer.as_default():
                    for unet_idx in range(len(imagen.unets)):
                        out_path = f"{out_dir}/sample-{epoch}-unet-{unet_idx}.png"
                        save_image(samples[unet_idx], str(out_path), nrow=10)
                        name = f"samples/unet_{unet_idx}" if unet_idx > 0 else "samples"
                        tf.summary.image(
                            name,
                            samples[unet_idx]
                            .cpu()
                            .detach()
                            .numpy()
                            .transpose(0, 2, 3, 1),
                            max_outputs=max_outputs,
                            step=epoch,
                        )
            if do_debug and cfg.train_one_epoch:
                break

    if not do_debug:
        torch.save(imagen.state_dict(), f"{out_dir}/imagen_epoch_{cfg.num_epochs}.pt")


def log_batch_input(eval_batch, epoch, max_outputs, prefix=None):
    train_img_name = "train_img"
    sdm_name = "sdm"
    rgb_name = "rgb"
    if prefix is not None:
        train_img_name = f"{prefix}/{train_img_name}"
        sdm_name = f"{prefix}/{sdm_name}"
        rgb_name = f"{prefix}/{rgb_name}"

    tf.summary.image(
        train_img_name,
        eval_batch["image"].numpy().transpose(0, 2, 3, 1),
        max_outputs=max_outputs,
        step=epoch,
    )
    tf.summary.image(
        sdm_name,
        eval_batch["sdm"].numpy().transpose(0, 2, 3, 1),
        max_outputs=max_outputs,
        step=epoch,
    )
    tf.summary.image(
        rgb_name,
        eval_batch["rgb"].numpy().transpose(0, 2, 3, 1),
        max_outputs=max_outputs,
        step=epoch,
    )
