import os
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from dataset.buildingnet_dataset import BuildingNetDataset, Split


class BuildingNetDataModule(pl.LightningDataModule):
    train_dataset: BuildingNetDataset | None = None
    val_dataset: BuildingNetDataset | None = None
    test_dataset: BuildingNetDataset | None = None

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
