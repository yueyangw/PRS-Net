import numpy.random
import torch
import numpy as np
from data.data_loader import get_loader
from model.model import Model
import matplotlib.pyplot as plt
from plot_result import plot_plane

fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(0, 32, 32)
y = np.linspace(0, 32, 32)
X, Y = np.meshgrid(x, y)

numpy.random.seed(1)
torch.manual_seed(1)


def train(epoch, model, learning_rate, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = 9999999
    for e in range(epoch):
        now_loss = 0
        id = 1
        batch_num = 0
        validate_loss = 0
        validate_num = 0
        while True:

            train_loader, validate_loader = get_loader(id, 32, device)
            id += 1
            if train_loader is None:
                break

            # train process
            model.train()
            for i, (voxels, points, pretreatment) in enumerate(train_loader):
                optimizer.zero_grad()

                voxels, points, pretreatment = voxels.to(device), points.to(device), pretreatment.to(device)

                planes, loss = model(voxels, points, pretreatment)

                loss.backward()
                optimizer.step()
                now_loss += loss.data
                batch_num += 1

            # validate process
            model.eval()
            for i, (voxels, points, pretreatment) in enumerate(validate_loader):
                voxels, points, pretreatment = voxels.to(device), points.to(device), pretreatment.to(device)

                with torch.no_grad():
                    planes, loss = model(voxels, points, pretreatment)

                if epoch == e + 1:
                    planes[0] = planes[0].to('cpu')
                    planes[1] = planes[1].to('cpu')
                    planes[2] = planes[2].to('cpu')
                    points = points.to('cpu')
                    planes = np.stack(planes)
                    for batch in range(planes.shape[1]):
                        plot_plane(planes[:, batch, :], points[batch].numpy(), (-0.5, 0.5))

                validate_loss += loss.data
                validate_num += 1

        if validate_loss / validate_num < min_loss:
            torch.save(model.state_dict(), 'model_weights.pth')
            min_loss = validate_loss / validate_num

        print("epoch: {}, t_loss: {}, v_loss: {}, min_loss: {}".format(e + 1, now_loss / batch_num, validate_loss / validate_num, min_loss))


if __name__ == '__main__':
    device = 'cpu'
    model = Model(device=device)
    train(60, model, 0.01, device)
