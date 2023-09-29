import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

data = {}


def get_source_data(id):
    if id not in data:
        try:
            points = np.load('data/processed/{}_points.npy'.format(id))
            pretreatment = np.load('data/processed/{}_pre.npy'.format(id))
            voxel = np.load('data/processed/{}_voxel.npy'.format(id))
        except:
            return None
        data[id] = {
            'points': points,
            'pretreatment': pretreatment,
            'voxel': voxel
        }

    return data[id]


class MyDataset(Dataset):

    def __init__(self, id, device):
        source_data = get_source_data(id)
        if source_data is None:
            self.none = True
            return

        self.points = torch.from_numpy(source_data['points']).to(device)
        self.pretreatment = torch.from_numpy(source_data['pretreatment']).to(device)
        self.voxel = torch.from_numpy(source_data['voxel']).to(device)

    def isEffective(self):
        try:
            if self.none:
                return False
            else:
                return True
        except:
            return True

    def __len__(self):
        return self.points.size(0)

    def __getitem__(self, item):
        return self.voxel[item], self.points[item], self.pretreatment[item]


def get_loader(id: int, batch_size: int = 32, device='cpu'):
    dataset = MyDataset(id, device)
    if not dataset.isEffective():
        return None, None

    length = len(dataset)
    train_size = int(0.8 * length)
    validate_size = length - train_size

    train_set, validate_set = random_split(dataset, [train_size, validate_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=True)

    return train_loader, validate_loader


if __name__ == '__main__':
    x_loader, _ = get_loader(1)
    for i, (a, b, c) in enumerate(x_loader):
        print(a.shape, b.shape, c.shape)
