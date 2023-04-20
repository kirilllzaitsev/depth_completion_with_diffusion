import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


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
