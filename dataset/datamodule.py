from typing import Optional
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .shapenet_dataset import Field, Shapes3dDataset

class ShapeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        fields: dict[str, Field],
        categories: Optional[list[str]],
        batch_size: int = 32,
        test_batch_size: int = 32,
        num_points: int = 2025,
        num_sdf_points: int = 5000,
        test_num_sdf_points: int = 5000,
        sampling_type: Optional[str] = None,
        collate_fn=None,
    ) -> None:
        super().__init__()

        assert (
            dataset_name == "Shapenet"
        ), "Only the ShapeNet dataset is currently supported."

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.categories = categories
        self.num_points = num_points
        self.num_sdf_points = num_sdf_points
        self.test_num_sdf_points = test_num_sdf_points
        self.sampling_type = sampling_type
        self.collate_fn = collate_fn
        self.fields = fields

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = Shapes3dDataset(
                self.dataset_path,
                self.fields,
                split="train",
                categories=self.categories,
                transform=None,
                num_points=self.num_points,
                num_sdf_points=self.num_sdf_points,
                sampling_type=self.sampling_type,
            )
            self.val_dataset = Shapes3dDataset(
                self.dataset_path,
                self.fields,
                split="val",
                categories=self.categories,
                transform=None,
                num_points=self.num_points,
                num_sdf_points=self.test_num_sdf_points,
                sampling_type=self.sampling_type,
            )
        elif stage == "test":
            self.test_dataset = Shapes3dDataset(
                self.dataset_path,
                self.fields,
                split="test",
                categories=self.categories,
                transform=None,
                num_points=self.num_points,
                num_sdf_points=self.test_num_sdf_points,
                sampling_type=self.sampling_type,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 0,
        )
