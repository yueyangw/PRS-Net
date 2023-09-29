import os

import numpy as np
import torch
import open3d as o3d

cates = os.listdir('ShapeNet')


def find_nearest_points(points, mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene.compute_closest_points(points)['points']


models = None
points_set = None
pretreatment_set = None
save_path = "processed/"

num_file = 0

for cate in cates:
    try:
        model_list = os.listdir('ShapeNet/' + cate)
    except:
        continue
    print(cate, end=' ')
    for model_id in model_list:
        path = 'ShapeNet/{}/{}/models/'.format(cate, model_id)
        if len(model_id) != 32:
            continue

        if models is not None:
            print(num_file+1, models.size(0))

        # sample点云
        try:
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, 'model_normalized.obj'))
            pcd = mesh.sample_points_uniformly(number_of_points=1000)
            points = torch.from_numpy(np.asarray(pcd.points))
        except:
            continue

        # 预训练最近点
        try:
            pretreatment = np.zeros([32, 32, 32, 3], dtype=np.float32)
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        pretreatment[i, j, k] = np.array([i, j, k])
            pretreatment = pretreatment / 32 + 1 / 64 - 0.5
            pretreatment = find_nearest_points(pretreatment, mesh)
            pretreatment = torch.from_numpy(pretreatment.numpy())
        except:
            continue

        # 提取体素
        try:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.03125)
            voxel_cells = torch.from_numpy(np.stack(list(vx.grid_index for vx in voxel_grid.get_voxels())))
            voxel_cells[:, 0] += ((points[:, 0].min() + 0.5) * 32).int()
            voxel_cells[:, 1] += ((points[:, 1].min() + 0.5) * 32).int()
            voxel_cells[:, 2] += ((points[:, 2].min() + 0.5) * 32).int()
            model = torch.zeros([32, 32, 32], dtype=torch.float32)
            for v in voxel_cells:
                try:
                    model[v[0], v[1], v[2]] = 1
                except:
                    continue
            model = model.unsqueeze(0)
        except:
            continue

        if pretreatment_set is None:
            pretreatment_set = pretreatment.unsqueeze(0)
        else:
            pretreatment_set = torch.cat([pretreatment_set, pretreatment.unsqueeze(0)], dim=0)

        if models is None:
            models = model.unsqueeze(0)
        else:
            models = torch.cat([models, model.unsqueeze(0)], 0)

        if points_set is None:
            points_set = points.unsqueeze(0)
        else:
            points_set = torch.cat([points_set, points.unsqueeze(0)], dim=0)

        size = models.size(0)
        if size == 2048:
            num_file += 1
            np.save(os.path.join(save_path, str(num_file)) + '_voxel', models.numpy())
            np.save(os.path.join(save_path, str(num_file)) + '_points', points_set.numpy())
            np.save(os.path.join(save_path, str(num_file)) + '_pre', pretreatment_set.numpy())
            models = None
            points_set = None
            pretreatment_set = None

num_file += 1
np.save(os.path.join(save_path, str(num_file)) + '_voxel', models.numpy())
np.save(os.path.join(save_path, str(num_file)) + '_points', points_set.numpy())
np.save(os.path.join(save_path, str(num_file)) + '_pre', pretreatment_set.numpy())
print(models.shape)
models = None
points_set = None
pretreatment_set = None
print('over!')
models = None
