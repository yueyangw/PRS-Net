import matplotlib.pyplot as plt
import numpy as np


def plot_plane(planes, voxel, scale=(0, 32)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(scale[0], scale[1], 100)
    y = np.linspace(scale[0], scale[1], 100)
    x, y = np.meshgrid(x, y)

    for plane in planes:
        a, b, c, d = plane
        z = (-a * x - b * y - d) / c

        mask = (z >= scale[0]) & (z <= scale[1])
        z[~mask] = np.nan

        ax.plot_surface(x, y, z, alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(scale[0], scale[1])
    ax.set_ylim(scale[0], scale[1])
    ax.set_zlim(scale[0], scale[1])

    plt.show()


if __name__ == '__main__':

    plane = [(3, 3, 3, 3), (1, 2, 3, 4)]

    plot_plane(plane, (-10, 10))
