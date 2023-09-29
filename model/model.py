import torch
from torch import nn
from model.loss import LossFunc


class Model(nn.Module):
    def __init__(
        self,
        in_channels=1,
        voxel_size=32,
        kernel_size=3,
        plane_num=3,
        neg_slop=0.2,
        Wr=25,
        device='cpu'
    ):
        super(Model, self).__init__()

        self.model = ModelArch(in_channels, kernel_size, neg_slop, plane_num, device)
        self.loss_model = LossFunc(voxel_size, Wr, device)

    def forward(self, voxel_input, points_input, pretreatment):
        planes = self.model(voxel_input)
        loss_v = self.loss_model(points_input, pretreatment, voxel_input, planes)

        return planes, loss_v


class ModelArch(nn.Module):

    def __init__(self, in_channels=1, kernel_size=3, neg_slop=0.2, plane_num=3, device='cpu'):
        super(ModelArch, self).__init__()

        self.plane_num = plane_num
        self.device = device

        # CNN网络结构
        channels = 4
        cnn_seq = [nn.Conv3d(in_channels, channels, kernel_size, 1, 1, device=device)]
        for i in range(4):
            cnn_seq.append(nn.MaxPool3d(2))
            cnn_seq.append(nn.LeakyReLU(neg_slop))
            cnn_seq.append(nn.Conv3d(channels, channels * 2, kernel_size, 1, 1, device=device))
            channels *= 2
        cnn_seq.append(nn.MaxPool3d(2))
        cnn_seq.append(nn.LeakyReLU(neg_slop))
        self.cnn_model = nn.Sequential(*cnn_seq)

        # 全连接层网络结构
        self.fnn_plane_models = []
        for i in range(plane_num):
            self.fnn_plane_models.append(nn.Sequential(
                nn.Linear(channels, channels // 2, device=device),
                nn.LeakyReLU(neg_slop),
                nn.Linear(channels // 2, channels // 4, device=device),
                nn.LeakyReLU(neg_slop),
                nn.Linear(channels // 4, 4, device=device),
            ))

    '''
    input: [batch, x, y, z]
    '''
    def forward(self, input):
        embedding = self.cnn_model(input)
        embedding = embedding.squeeze()

        planes = []
        for i in range(self.plane_num):
            plane = self.fnn_plane_models[i](embedding)
            plane = nn.functional.normalize(plane, p=2, dim=1)
            planes.append(plane)

        return planes


if __name__ == '__main__':
    model = Model()
    voxel_input = torch.zeros([2, 1, 32, 32, 32])
    points_input = torch.rand([2, 1000, 3])
    points_input = points_input / torch.max(points_input) - 0.5
    pretreatment = torch.rand([2, 32, 32, 32, 3])
    p, r, l = model(voxel_input, points_input, pretreatment)
    print(l)
