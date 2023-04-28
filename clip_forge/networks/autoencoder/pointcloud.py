import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet_Head(nn.Module):
    def __init__(self, pc_dims=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, pc_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(pc_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return x


class PointNet(nn.Module):
    def __init__(self, c_dim=512, pc_dims=1024, last_feature_transform=None):
        super(PointNet, self).__init__()
        self.point_head = PointNet_Head(pc_dims=pc_dims)
        self.projection_layer = nn.Sequential(
            nn.Linear(pc_dims, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, c_dim),
        )
        self.last_feature_transform = last_feature_transform

    def forward(self, x):
        x = self.point_head(x)
        x = self.projection_layer(x)

        if self.last_feature_transform == "add_noise" and self.training is True:
            x = x + 0.1 * torch.randn(*x.size()).to(x.device)

        return x


## borrowed from https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti
def GridSamplingLayer(batch_size, meshgrid):
    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
    return g


class Foldingnet_decoder(nn.Module):
    def __init__(self, num_points, z_dim):
        super().__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(z_dim + 2, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 1)
        self.conv4 = torch.nn.Conv1d(z_dim + 3, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 3, 1)

        self.act = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(3)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(512)

        # self.device = args.device

    def forward(self, x):
        meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        grid = GridSamplingLayer(x.size(0), meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid).to(x.device)
        grid = grid.transpose(-1, 1)

        latent = x = (
            x.unsqueeze(2).expand(x.size(0), x.size(1), grid.size(2)).contiguous()
        )

        x = torch.cat((grid, x), 1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        x = torch.cat((x, latent), 1).contiguous()
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x.contiguous().transpose(2, 1).contiguous()
