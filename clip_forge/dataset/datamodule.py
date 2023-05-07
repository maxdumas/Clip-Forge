import os
from pathlib import Path
from typing import Optional

from clip.model import CLIP
import clip
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np

from .buildingnet_dataset import BuildingNetDataset, Split
from ..networks.autoencoder import Autoencoder
from ..utils.helper import InputType


class BuildingNetDataModule(pl.LightningDataModule):
    train_dataset: Optional[BuildingNetDataset] = None
    val_dataset: Optional[BuildingNetDataset] = None
    test_dataset: Optional[BuildingNetDataset] = None

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Path,
        batch_size: int = 32,
        test_batch_size: int = 32,
        num_points: int = 2025,
        num_sdf_points: int = 5000,
        test_num_sdf_points: int = 5000,
        image_resolution: int = 224,
    ):
        super().__init__()

        assert (
            dataset_name == "BuildingNet"
        ), "Only the BuildingNet dataset is currently supported."

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_points = num_points
        self.num_sdf_points = num_sdf_points
        self.test_num_sdf_points = test_num_sdf_points
        self.image_resolution = image_resolution

    def setup(self, stage: str) -> None:
        base_args = {
            "dataset_root": self.dataset_path,
            "num_sdf_points": self.num_sdf_points,
            "num_pc_points": self.num_points,
            "image_resolution": self.image_resolution,
        }
        if stage == "fit":
            self.train_dataset = BuildingNetDataset(**base_args, split=Split.TRAIN)
            self.val_dataset = BuildingNetDataset(**base_args, split=Split.VAL)
        elif stage == "test":
            self.test_dataset = BuildingNetDataset(**base_args, split=Split.TEST)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=os.cpu_count() or 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
            pin_memory=torch.cuda.is_available(),
        )


clip_models = {
    "B-16": {
        "name": "ViT-B/16",
        "emb_dim": 512,
    },
    "B-32": {
        "name": "ViT-B/32",
        "emb_dim": 512,
    },
    "RN50x16": {
        "name": "RN50x16",
        "emb_dim": 768,
    },
}


def get_clip_model(clip_model_type: str, device) -> tuple[CLIP, int, int]:
    model_opts = clip_models[clip_model_type]
    clip_model, _ = clip.load(model_opts["name"], device=device)
    input_resolution = clip_model.visual.input_resolution
    return clip_model, input_resolution, model_opts["emb_dim"]


class BuildingNetEmbeddingDataModule(BuildingNetDataModule):
    clip_model: CLIP
    autoencoder: Autoencoder
    device: str
    n_embeddings_per_datum: int
    image_resolution: int
    cond_emb_dim: int

    def __init__(
        self,
        device: str,
        autoencoder_checkpoint_path: str,
        clip_model_type: str,
        input_type: InputType,
        dataset_name: str,
        dataset_path: Path,
        n_embeddings_per_datum=5,
        batch_size: int = 32,
        test_batch_size: int = 32,
        num_points: int = 2025,
        num_sdf_points: int = 5000,
        test_num_sdf_points: int = 5000,
    ):
        # Load CLIP
        self.clip_model, self.image_resolution, self.cond_emb_dim = get_clip_model(
            clip_model_type, device
        )
        self.input_type = input_type

        super().__init__(
            dataset_name,
            dataset_path,
            batch_size,
            test_batch_size,
            num_points,
            num_sdf_points,
            test_num_sdf_points,
            self.image_resolution,
        )

        self.device = device
        self.n_embeddings_per_datum = n_embeddings_per_datum

        print(
            f"Loading specified W&B autoencoder Checkpoint from {autoencoder_checkpoint_path}."
        )
        self.autoencoder = Autoencoder.load_from_checkpoint(
            autoencoder_checkpoint_path
        ).to(
            device
        )  # TODO: Load device in a better way

    def get_condition_embeddings(
        self,
        dataloader: DataLoader,
    ):
        """
        Given an Autoencoder and CLIP model, generates 3D shape embeddings using the
        Autoencoder and 2D image rendering embeddings using CLIP for every data
        point in the dataset represented by dataloader.

        Note that this implies the dataset must have available 2D image renderings
        of the 3D model.

        Embeddings will be repeatedly generated `n_embeddings_per_datum` times.
        """
        self.autoencoder.eval()
        self.clip_model.eval()
        shape_embeddings = []
        cond_embeddings = []
        with torch.no_grad():
            for _ in range(0, self.n_embeddings_per_datum):
                for data in tqdm(dataloader):
                    image = data["images"].type(torch.FloatTensor).to(self.device)

                    if self.input_type == InputType.VOXELS:
                        data_input = (
                            data["voxels"].type(torch.FloatTensor).to(self.device)
                        )
                    elif self.input_type == InputType.POINTCLOUD:
                        data_input = (
                            data["pc_org"]
                            .type(torch.FloatTensor)
                            .to(self.device)
                            .transpose(-1, 1)
                        )

                    shape_emb = self.autoencoder.net.reparameterize(*self.autoencoder.net.encoder(data_input))

                    image_features = self.clip_model.encode_image(image)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )

                    shape_embeddings.append(shape_emb.detach().cpu().numpy())
                    cond_embeddings.append(image_features.detach().cpu().numpy())

            shape_embeddings = np.concatenate(shape_embeddings)
            cond_embeddings = np.concatenate(cond_embeddings)

        return torch.utils.data.TensorDataset(
            torch.from_numpy(shape_embeddings),
            torch.from_numpy(cond_embeddings),
        )

    def setup(self, stage: str) -> None:
        # Setup normal BuildingNet datasets
        super().setup(stage)
        # Transform these datasets into the conditional embedding datasets
        if stage == "fit":
            self.train_dataset = self.get_condition_embeddings(
                super().train_dataloader()
            )
            self.val_dataset = self.get_condition_embeddings(super().val_dataloader())
        else:
            self.test_dataset = self.get_condition_embeddings(super().test_dataloader())
