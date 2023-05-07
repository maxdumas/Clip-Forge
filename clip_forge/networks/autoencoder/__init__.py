from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...utils.helper import InputType, OutputType
from ...utils.visualization import make_3d_grid
from .implicit import Occ_Simple_Decoder, VoxelEncoderBN
from .implicit_vae import VoxelVariationalAutoencoder
from .pointcloud import Foldingnet_decoder, PointNet


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

        assert self.input_type == InputType.VOXELS
        assert self.output_type == OutputType.IMPLICIT

        self.net = VoxelVariationalAutoencoder(self.emb_dims)

    def reconstruction_loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        return F.mse_loss(pred.squeeze(-1), gt)

    def forward(self, data_input, query_points=None) -> Tensor:
        return self.net.forward(data_input, query_points)

    def configure_optimizers(self):
        # TODO: How do we parameterize this using LightningCLI?
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def extract_input(self, data: dict) -> Tensor:
        # Load appropriate input data from the training set
        return data["voxels"]

    def extract_ground_truth(self, data: dict) -> tuple[Tensor, Tensor]:
        return data["points_occ"], data["points"]

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
        pred = self.forward(data_input, query_points)

        # Compute reconstruction loss
        loss = self.reconstruction_loss(pred, gt)

        self.log("loss/train/reconstruction", loss, prog_bar=True)

        return loss

    def validation_step(self, data: dict, data_idx):
        data_input = self.extract_input(data)
        gt, query_points = self.extract_ground_truth(data)

        pred = self.forward(data_input, query_points)

        loss = self.reconstruction_loss(pred, gt)

        self.log("loss/val/reconstruction", loss)

        if self.output_type == OutputType.IMPLICIT:
            # Compute IOU loss for Implicit representation
            query_points = self.create_sampling_grid(data_input)

            # Run prediction
            pred = self.forward(data_input, query_points)

            voxels_occ_np = data["voxels"] >= 0.5
            occ_hat_np = pred >= self.threshold
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()
            loss = iou_voxels.item()
            self.log("loss/val/iou", loss, prog_bar=True)

        return pred

    def test_step(self, data: dict, data_idx):
        data_input = self.extract_input(data)
        gt, query_points = self.extract_ground_truth(data)

        pred = self.forward(data_input, query_points)

        loss = self.reconstruction_loss(pred, gt)

        self.log("loss/test/reconstruction", loss)

        if self.output_type == OutputType.IMPLICIT:
            # Compute IOU loss for Implicit representation
            query_points = self.create_sampling_grid(data_input)

            # Run prediction
            pred = self.forward(data_input, query_points)

            voxels_occ_np = data["voxels"] >= 0.5
            occ_hat_np = pred >= self.threshold
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()
            loss = iou_voxels.item()
            self.log("loss/test/iou", loss, prog_bar=True)

        return pred
