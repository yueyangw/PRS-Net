# from mpl_toolkits import mplot3d
import numpy as np
import data.binvox_rw as binvox
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
fig = plt.figure()
ax = plt.axes(projection='3d')

with open('../data/ShapeNet/02773838/8bc53bae41a8105f5c7506815f553527/models/model_normalized.solid.binvox', 'rb') as f:
    model = binvox.read_as_3d_array(f)

# with open('data/04379243/expert_verified/points_label/240ddf8b63318ef534506cc3910614fe.seg') as ff:
#     cate = ff.readlines()

ts = torch.Tensor(model.data)
ts = torch.unsqueeze(ts, 0)
ts = torch.unsqueeze(ts, 0)

x = ts
# x = F.interpolate(ts, scale_factor=0.25, mode='trilinear', align_corners=False)
print(x.shape)
ax.voxels(x[0, 0])
ax.set_title('3d Scatter plot')
plt.show()