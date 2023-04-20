from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.helper import InputType, OutputType
from ..utils.visualization import make_3d_grid
from .network_utils import ResnetBlockFC

EPS = 1e-6


# borrowed from https://github.com/ThibaultGROUEIX/AtlasNet
def dist_chamfer(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


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


class VoxelEncoderBN(nn.Module):
    def __init__(self, dim=3, c_dim=128, last_feature_transform=None):
        super().__init__()

        self.net = nn.Sequential(
            # Input convolution
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Hidden convolution 1
            nn.Conv3d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Hidden convolution 2
            nn.Conv3d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            # Hidden convolution 3
            nn.Conv3d(128, 256, 3, padding=1, stride=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            # Output convolution
            nn.Conv3d(256, 512, 3, padding=1, stride=2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(512 * 2 * 2 * 2, c_dim),
        )

        self.last_feature_transform = last_feature_transform

    def forward(self, x: Tensor):
        x = x.unsqueeze(1)
        x = self.net(x)

        if self.last_feature_transform == "add_noise" and self.training is True:
            x = x + 0.1 * torch.randn(*x.size(), device=x.device)

        return x


class Occ_Simple_Decoder(nn.Module):
    def __init__(
        self, z_dim=128, point_dim=3, hidden_size=128, leaky=False, last_sig=False
    ):
        super().__init__()
        self.z_dim = z_dim

        # Submodules
        self.fc_p = nn.Linear(point_dim, hidden_size)
        self.fc_z = nn.Linear(z_dim, hidden_size)

        self.net = nn.Sequential(
            ResnetBlockFC(hidden_size),
            ResnetBlockFC(hidden_size),
            ResnetBlockFC(hidden_size),
            ResnetBlockFC(hidden_size),
            ResnetBlockFC(hidden_size),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.last_sig = last_sig

    def forward(self, p: Tensor, z: Tensor):
        net_p = self.fc_p(p)
        net_z = self.fc_z(z).unsqueeze(1)

        out = self.net(net_p + net_z)
        out = out.squeeze(-1)

        if self.last_sig == True:
            out = torch.sigmoid(out)

        return out


def compute_iou(occ1: Tensor, occ2: Tensor):
    """Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    """
    a: np.ndarray = occ1.cpu().numpy()
    b: np.ndarray = occ2.cpu().numpy()

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if a.ndim >= 2:
        a = a.reshape(a.shape[0], -1)
    if b.ndim >= 2:
        b = b.reshape(b.shape[0], -1)

    # Convert to boolean values
    a = a >= 0.5
    b = b >= 0.5

    # Compute IOU
    area_union = (a | b).astype(np.float32).sum(axis=-1)
    area_intersect = (a & b).astype(np.float32).sum(axis=-1)

    iou = area_intersect / area_union

    return iou


class Autoencoder(pl.LightningModule):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        batch_size: int = 32,
        test_batch_size: int = 32,
        lr: float = 0.0001,
        num_points: int = 2025,
        emb_dims: int = 256,
        input_type: InputType = InputType.VOXELS,
        output_type: OutputType = OutputType.IMPLICIT,
        threshold: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        ### Local Model Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.emb_dims = emb_dims
        self.input_type = input_type
        self.output_type = output_type
        self.threshold = threshold

        ### Sub-Network def
        if self.input_type == InputType.VOXELS:
            self.encoder = VoxelEncoderBN(
                dim=3, c_dim=emb_dims, last_feature_transform="add_noise"
            )
        elif self.input_type == InputType.POINTCLOUD:
            self.encoder = PointNet(
                pc_dims=1024, c_dim=emb_dims, last_feature_transform="add_noise"
            )

        if self.output_type == OutputType.IMPLICIT:
            self.decoder = Occ_Simple_Decoder(z_dim=emb_dims)
        elif self.output_type == OutputType.POINTCLOUD:
            self.decoder = Foldingnet_decoder(num_points=num_points, z_dim=emb_dims)

    def decoding(self, shape_embedding, points=None):
        if self.output_type == OutputType.IMPLICIT:
            return self.decoder(points, shape_embedding)
        elif self.output_type == OutputType.POINTCLOUD:
            return self.decoder(shape_embedding)

    def reconstruction_loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        if self.output_type == OutputType.IMPLICIT:
            return F.mse_loss(pred.squeeze(-1), gt)
        elif self.output_type == OutputType.POINTCLOUD:
            dl, dr = dist_chamfer(gt, pred)
            return (dl.mean(dim=1) + dr.mean(dim=1)).mean()

    def forward(self, data_input, query_points=None) -> tuple[Tensor, Tensor]:
        shape_embs = self.encoder(data_input)
        pred = self.decoding(shape_embs, points=query_points)
        return pred, shape_embs

    def configure_optimizers(self):
        # TODO: How do we parameterize this using LightningCLI?
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def extract_input(self, data: dict) -> Tensor:
        # Load appropriate input data from the training set
        if self.input_type == InputType.VOXELS:
            return data["voxels"]
        elif self.input_type == InputType.POINTCLOUD:
            return data["pc_org"].transpose(-1, 1)

    def extract_ground_truth(self, data: dict) -> tuple[Tensor, Optional[Tensor]]:
        if self.output_type ==  OutputType.IMPLICIT:
            return data["points_occ"], data["points"]
        elif self.output_type == OutputType.POINTCLOUD:
            return data["pc_org"], None

    def create_sampling_grid(self, data_input: Tensor) -> Tensor:
        points_voxels = make_3d_grid(
            (-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, (32,) * 3, device=self.device
        )
        query_points = points_voxels.expand(self.test_batch_size, *points_voxels.size())

        if self.test_batch_size != data_input.size(0):
            query_points = points_voxels.expand(
                data_input.size(0), *points_voxels.size()
            )
        return query_points

    def training_step(self, data: dict, batch_idx):
        data_input = self.extract_input(data)
        gt, query_points = self.extract_ground_truth(data)

        # Run prediction
        pred, _ = self.forward(data_input, query_points)

        # Compute reconstruction loss
        loss = self.reconstruction_loss(pred, gt)

        self.log("loss/train/reconstruction", loss, prog_bar=True)

        return loss

    def validation_step(self, data: dict, data_idx):
        data_input = self.extract_input(data)
        gt, query_points = self.extract_ground_truth(data)

        pred, _ = self.forward(data_input, query_points)

        loss = self.reconstruction_loss(pred, gt)

        self.log("loss/val/reconstruction", loss)

        if self.output_type == OutputType.IMPLICIT:
            # Compute IOU loss for Implicit representation
            query_points = self.create_sampling_grid(data_input)

            # Run prediction
            pred, _ = self.forward(data_input, query_points)

            voxels_occ_np = data["voxels"] >= 0.5
            occ_hat_np = pred >= self.threshold
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()
            loss = iou_voxels.item()
            self.log("loss/val/iou", loss, prog_bar=True)

        return pred

    def test_step(self, data: dict, data_idx):
        data_input = self.extract_input(data)
        gt, query_points = self.extract_ground_truth(data)

        pred, _ = self.forward(data_input, query_points)

        loss = self.reconstruction_loss(pred, gt)

        self.log("loss/test/reconstruction", loss)

        if self.output_type == OutputType.IMPLICIT:
            # Compute IOU loss for Implicit representation
            query_points = self.create_sampling_grid(data_input)

            # Run prediction
            pred, _ = self.forward(data_input, query_points)

            voxels_occ_np = data["voxels"] >= 0.5
            occ_hat_np = pred >= self.threshold
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()
            loss = iou_voxels.item()
            self.log("loss/test/iou", loss, prog_bar=True)

        return pred