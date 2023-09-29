import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import get_loader

voxel = np.load('processed/1_voxel.npy')
points = np.load('processed/1_points.npy')
voxel = torch.where(torch.from_numpy(voxel)>0, 1, 0)
points = torch.from_numpy(points)
points = (points + 0.5) * 32
points = points.int()

fig = plt.figure()
ax = plt.axes(projection='3d')

plt.xlim((0, 32))
plt.ylim((0, 32))
ax.set_zlim((0, 32))


plt.ion()
model_idx = 3
ax.voxels(voxel[model_idx, 0])
ax.scatter3D(points[model_idx, :, 0], points[model_idx, :, 1], points[model_idx, :, 2])
ax.set_title('3d Scatter plot')
plt.ioff()
plt.show()
