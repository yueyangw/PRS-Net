import math

import matplotlib.pyplot as plt
import numpy as np


def calc_angle(a, b):
    a = a[0:3]
    b = b[0:3]
    angle = math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return angle


img_num = 0


def plot_plane(planes, points, scale=(0, 32)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(scale[0], scale[1], 2000)
    y = np.linspace(scale[0], scale[1], 2000)
    x, y = np.meshgrid(x, y)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=1)

    if_plot = [1, 1, 1]
    for i in range(planes.shape[0]):
        for j in range(i+1, planes.shape[0]):
            if calc_angle(planes[i], planes[j]) < math.pi / 6:
                if_plot[j] = 0

    for i in range(planes.shape[0]):
        if if_plot[i] == 0:
            continue
        plane = planes[i]
        a, b, c, d = plane
        z = (-a * x - b * y - d) / c

        mask = (z >= scale[0]) & (z <= scale[1])
        z[~mask] = np.nan

        ax.plot_surface(x, y, z, alpha=0.3, color='g')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.axis('off')
    ax.view_init(elev=45, azim=-60)

    ax.set_xlim(scale[0], scale[1])
    ax.set_ylim(scale[0], scale[1])
    ax.set_zlim(scale[0], scale[1])

    global img_num
    img_num += 1
    plt.savefig('result/' + str(img_num) + '.png')
    # plt.show()


if __name__ == '__main__':
    plane = [(3, 3, 3, 3), (1, 2, 3, 4)]

    plot_plane(plane, (-10, 10))
