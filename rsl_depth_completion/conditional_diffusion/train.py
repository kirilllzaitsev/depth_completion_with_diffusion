import torch
from tqdm import tqdm
import numpy as np


def MinimagenTrain(
    timestamp,
    args,
    unets,
    imagen,
    train_dataloader,
    valid_dataloader,
    training_dir,
    optimizer,
):
    """
    Training loop for MinImagen instance

    :param timestamp: Timestamp for training.
    :param args: Arguments Namespace/dict from argparsing :func:`.minimagen.training.get_minimagen_parser` parser.
    :param unets: List of :class:`~.minimagen.Unet.Unet`s used in the Imagen instance.
    :param imagen: :class:`~.minimagen.Imagen.Imagen` instance to train.
    :param train_dataloader: Dataloader for training.
    :param valid_dataloader: Dataloader for validation.
    :param training_dir: Training directory context manager returned from :func:`~.minimagen.training.create_directory`.
    :param optimizer: Optimizer to use for training.
    :param timeout: Amount of time to spend trying to process batch before passing on to the next batch. Does not work
        on Windows.
    :return:
    """
    best_loss = [torch.tensor(9999999) for i in range(len(unets))]
    train_unet_losses = {"base": [], "super": []}
    val_unet_losses = {"base": [], "super": []}

    for epoch in range(args.EPOCHS):
        print(f"### Epoch {epoch + 1} of {args.EPOCHS} ###")

        imagen.train(True)

        running_train_loss = [0.0 for i in range(len(unets))]
        for batch_num, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="train",
        ):
            optimizer.zero_grad()
            if batch is None:
                print(f"Batch {batch_num} is None, skipping...")
                continue
            images = batch["image"]
            encoding = batch["encoding"]
            mask = batch["mask"]

            losses = [0.0 for i in range(len(unets))]
            for unet_idx in range(len(unets)):
                loss = imagen(
                    images,
                    text_embeds=encoding,
                    text_masks=mask,
                    unet_number=unet_idx + 1,
                )
                losses[unet_idx] = loss.detach()
                running_train_loss[unet_idx] += loss.detach()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(imagen.parameters(), 50)

            optimizer.step()

        avg_loss = [
            round(i.item() / len(train_dataloader), 5) for i in running_train_loss
        ]
        print(f"(Base,SuperR) Unets Train Loss: {avg_loss[0], avg_loss[1]}")
        train_unet_losses["base"].append(avg_loss[0])
        train_unet_losses["super"].append(avg_loss[1])

        # Compute average loss across validation batches for each unet
        running_valid_loss = [0.0 for i in range(len(unets))]
        imagen.train(False)

        for vbatch in tqdm(
            valid_dataloader,
            desc="valid",
            total=len(valid_dataloader),
        ):
            if not vbatch:
                continue

            images = vbatch["image"]
            encoding = vbatch["encoding"]
            mask = vbatch["mask"]

            for unet_idx in range(len(unets)):
                running_valid_loss[unet_idx] += imagen(
                    images,
                    text_embeds=encoding,
                    text_masks=mask,
                    unet_number=unet_idx + 1,
                ).detach()

        # Write average validation loss
        avg_val_loss = [
            round(i.item() / len(valid_dataloader), 5) for i in running_valid_loss
        ]

        # If validation loss less than previous best, save the model weights
        for i, l in enumerate(avg_val_loss):
            if l < best_loss[i]:
                best_loss[i] = l
                with training_dir("state_dicts"):
                    model_path = f"unet_{i}_state_{timestamp}.pth"
                    torch.save(imagen.unets[i].state_dict(), model_path)

        print(f"(Base,SuperR) Unets Val Loss: {avg_val_loss[0], avg_val_loss[1]}")
        val_unet_losses["base"].append(avg_val_loss[0])
        val_unet_losses["super"].append(avg_val_loss[1])

    for losses in [train_unet_losses, val_unet_losses]:
        losses["base"] = np.array(losses["base"])
        losses["super"] = np.array(losses["super"])
    return train_unet_losses, val_unet_losses
