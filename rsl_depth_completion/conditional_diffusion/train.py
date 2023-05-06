import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_depth_completion.conditional_diffusion.imagen import Imagen
from tqdm.auto import tqdm


def MinimagenTrain(
    timestamp,
    args,
    unets,
    imagen: Imagen,
    train_dataloader,
    valid_dataloader,
    training_dir,
    optimizer,
    **kwargs,
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
    train_unet_losses = {"base": [], "super": []}
    val_unet_losses = {"base": [], "super": []}
    start_epoch = kwargs.get("start_epoch", 0)

    for epoch in range(start_epoch, start_epoch + args.EPOCHS):
        print(f"### Epoch {epoch + 1} of {args.EPOCHS+start_epoch} ###")

        imagen.train(True)

        running_train_loss = [0.0 for i in range(len(unets))]
        for batch_num, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="train",
            leave=True,
        ):
            optimizer.zero_grad()
            if batch is None:
                print(f"Batch {batch_num} is None, skipping...")
                continue
            images = batch["image"]
            encoding = batch["encoding"]
            cond_image = batch["cond_image"]
            mask = batch["mask"]

            losses = [0.0 for i in range(len(unets))]
            for unet_idx in range(len(unets)):
                forward_out = imagen(
                    images,
                    text_embeds=encoding,
                    text_masks=mask,
                    unet_number=unet_idx + 1,
                )
                loss = forward_out["loss"]
                pred, noise = forward_out["pred"], forward_out["noise"]
                losses[unet_idx] = loss.detach()
                running_train_loss[unet_idx] += loss.detach()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(imagen.parameters(), 50)

            optimizer.step()

            if epoch % 10 == 0:
                # if batch_num % 100 == 0:
                # if batch_num % 100 == 0 and epoch % 2 == 0:
                avg_running_loss = [
                    round(i.item() / (batch_num + 1), 5) for i in running_train_loss
                ]
                print(f"Running Train Losses at batch {batch_num}: {avg_running_loss}")
                sample_args = {
                    "cond_scale": 3.0,
                    "timesteps": 200,
                    "return_last": False,
                }
                sample_out = imagen.sample(
                    texts=images[0],
                    text_masks=mask[0].unsqueeze(0),
                    text_embeds=encoding[0].unsqueeze(0),
                    return_pil_images=False,
                    **sample_args,
                )
                # import ipdb; ipdb.set_trace()
                from torchvision.utils import save_image

                save_image(
                    sample_out['base'][0].cpu(),
                    str(
                        f'{kwargs["save_train_samples_dir"]}/sample_epoch_{epoch}_batch_{batch_num}.png'
                    ),
                    n_row=10,
                )
                if kwargs.get("save_input") is not None and epoch < start_epoch+30:
                    save_image(
                        cond_image / 255,
                        str(f'{kwargs["save_train_samples_dir"]}/cond-input-{epoch}.png'),
                    )
                    save_image(
                        images, str(f'{kwargs["save_train_samples_dir"]}/input-{epoch}.png')
                    )

        avg_loss = [
            round(i.item() / len(train_dataloader), 5) for i in running_train_loss
        ]
        if len(imagen.unets) > 1:
            print(f"(Base,SuperR) Unets Train Loss: {avg_loss[0], avg_loss[1]}")
            train_unet_losses["super"].append(avg_loss[1])
        else:
            print(f"(Base) Unet Train Loss: {avg_loss[0]}")
        train_unet_losses["base"].append(avg_loss[0])

    return train_unet_losses, val_unet_losses

    #     # Compute average loss across validation batches for each unet
    #     running_valid_loss = [0.0 for i in range(len(unets))]
    #     imagen.train(False)

    #     for batch_num, vbatch in tqdm(
    #         enumerate(valid_dataloader),
    #         desc="valid",
    #         total=len(valid_dataloader),
    #     ):
    #         if not vbatch:
    #             continue

    #         images = vbatch["image"]
    #         encoding = vbatch["encoding"]
    #         mask = vbatch["mask"]

    #         for unet_idx in range(len(unets)):
    #             running_valid_loss[unet_idx] += imagen(
    #                 images,
    #                 text_embeds=encoding,
    #                 text_masks=mask,
    #                 unet_number=unet_idx + 1,
    #             ).detach()

    #         if batch_num % 100 == 0:
    #             avg_running_val_loss = [
    #                 round(i.item() / (batch_num + 1), 5) for i in running_valid_loss
    #             ]
    #             print(
    #                 f"Running Val Losses at batch {batch_num}: {avg_running_val_loss}"
    #             )

    #     # Write average validation loss
    #     avg_val_loss = [
    #         round(i.item() / len(valid_dataloader), 5) for i in running_valid_loss
    #     ]

    #     # If validation loss less than previous best, save the model weights
    #     for i, l in enumerate(avg_val_loss):
    #         if l < best_loss[i]:
    #             best_loss[i] = l
    #             with training_dir("state_dicts"):
    #                 model_path = f"unet_{i}_state_{timestamp}.pth"
    #                 torch.save(imagen.unets[i].state_dict(), model_path)

    #     print(f"(Base,SuperR) Unets Val Loss: {avg_val_loss[0], avg_val_loss[1]}")
    #     val_unet_losses["base"].append(avg_val_loss[0])
    #     val_unet_losses["super"].append(avg_val_loss[1])

    # for losses in [train_unet_losses, val_unet_losses]:
    #     losses["base"] = np.array(losses["base"])
    #     losses["super"] = np.array(losses["super"])
    # return train_unet_losses, val_unet_losses
