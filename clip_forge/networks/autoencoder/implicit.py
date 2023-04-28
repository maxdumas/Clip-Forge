import torch
import torch.nn as nn
from torch import Tensor

from ..network_utils import ResnetBlockFC

class SimpleAutoencoder(nn.Module):
    def __init__(self, emb_dims=128):
        super().__init__()

        self.encoder = VoxelEncoderBN(
            dim=3, c_dim=emb_dims, last_feature_transform=None
        )
        self.decoder = Occ_Simple_Decoder(z_dim=emb_dims)

    def forward(self, data_input, query_points=None) -> tuple[Tensor, Tensor]:
        shape_embs = self.encoder(data_input)
        pred = self.decoder(query_points, shape_embs)
        return pred, shape_embs


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
        self, z_dim=128, point_dim=3, hidden_size=128, leaky=True, last_sig=True
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
