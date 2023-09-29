import torch
from torch import nn


def get_distance(p1, p2):
    return nn.functional.pairwise_distance(p1, p2, p=2)


def tran_plane(plane, points):
    # num_points = points.size(1)
    norm_vector = plane[:, 0:3]
    len_norm = torch.norm(plane[:, 0:3], p=2, dim=1)
    points = points - torch.transpose(torch.unsqueeze((torch.sum(points.transpose(0, 1) * norm_vector, dim=2) + plane[:, 3]) / (len_norm ** 2), 2).repeat(
        1, 1, 3) * plane[:, 0:3] * 2, 0, 1)
    return points


class LossFunc(nn.Module):

    def __init__(self, size=32, Wr=25, device='cpu'):
        super(LossFunc, self).__init__()
        self.device = device
        self.size = size
        self.Wr = Wr

    '''
    sample_points: [batch, num, dim]
    planes: [ [batch, 4], ... , ]
    '''
    def forward(self, sample_points, pretreatment, voxel, planes):
        # pretreatment [batch, x, y, z, 3]
        # print(sample_points.shape)
        reg_loss = self.calc_reg_loss(planes)

        symLoss = torch.zeros([], device=self.device)
        for i in range(len(planes)):
            tran_points = tran_plane(planes[i], sample_points)
            symLoss += self.get_sym_loss(tran_points, pretreatment, voxel)

        return symLoss + self.Wr * reg_loss

    def get_point_cells(self, points):
        bound = 0.5
        res = points.view([points.size(0)*points.size(1), points.size(2)])
        res = (res + bound) * self.size
        res = res.view(points.shape).long()
        res = torch.where(res >= self.size, self.size - 1, res)
        res = torch.where(res < 0, 0, res)  # 限制界限
        return res  # [batch, point_num, 3]

    def get_sym_loss(self, points, pretreatment, voxel):
        batch = points.size(0)
        points_num = points.size(1)
        size = self.size
        idx = self.get_point_cells(points)
        g_idx = idx[:, :, 0] * size**2 + idx[:, :, 1] * size + idx[:, :, 2]
        useless = torch.gather(
            voxel.view(batch, size**3, -1),
            index=g_idx.view(batch, points_num, -1).long(),
            dim=1
        )
        useless = 1 - useless
        target_points = torch.gather(
            pretreatment.view(batch, size**3, -1),
            index=g_idx.view(batch, points_num, -1).repeat(1, 1, 3).long(),
            dim=1
        ).view(batch, points_num, 3)
        # print((1 - useless).sum()/batch/10)
        d = get_distance(points, target_points) * useless.squeeze()
        return torch.sum(d) / batch

    def calc_reg_loss(self, planes):
        batch = planes[0].size(0)
        m1 = torch.zeros([batch, len(planes), 3], device=self.device)
        for i in range(len(planes)):
            m1[:, i] = nn.functional.normalize(planes[i][:, 0:3], dim=1, p=2)
        f1 = torch.norm(m1 * torch.transpose(m1, 1, 2) - torch.eye(3, device=self.device), p=2) ** 2
        return f1 / batch