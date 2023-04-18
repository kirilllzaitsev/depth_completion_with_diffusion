import os
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


def animate_diffusion(samples, timesteps, valid_img_shape):
    random_index = np.random.randint(0, samples[0].shape[0])

    fig = plt.figure()
    ims = []
    for i in range(timesteps):
        # valid_img_shape = image_size, image_size, channels
        im = plt.imshow(
            samples[i][random_index].reshape(valid_img_shape),
            cmap="gray",
            animated=True,
        )
        ims.append([im])

    animate = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000
    )
    animate.save("diffusion.gif")
    plt.show()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
