from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path
from enum import Enum, auto
import random
from typing import Generic, TypeVar

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
    PILToTensor,
)

from .binvox_rw import read_as_3d_array


class Split(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    ALL = auto()


split_mapping: dict[Split, str] = {
    Split.TRAIN: "train_split.txt",
    Split.VAL: "val_split.txt",
    Split.TEST: "test_split.txt",
    Split.ALL: "dataset_models.txt",
}


@dataclass
class BuildingNetDatum:
    images: Tensor
    voxels: Tensor
    points: Tensor
    points_occ: Tensor
    # TODO: Load this field
    # pc_org: Tensor


TField = TypeVar("TField")


class Field(ABC, Generic[TField]):
    @abstractmethod
    def load(self, model_name: str) -> TField:
        pass

def to_rgb(image):
    return image.convert("RGB")

class ImagesField(Field[Tensor]):
    def __init__(self, root: Path, resolution: int, random_view=True):
        self.root = root
        self.random_view = random_view

        self.transform = Compose(
            [
                Resize(resolution, interpolation=Image.BICUBIC),
                CenterCrop(resolution),
                to_rgb, # TODO: Validate if we need this
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def load(self, model_name: str) -> Tensor:
        all_image_paths = list(self.root.glob(f"{model_name}*.png"))
        assert (
            len(all_image_paths) > 0
        ), f"No images found for {model_name} at {self.root.resolve(strict=True)}"

        if self.random_view:
            image_path = random.choice(all_image_paths)
        else:
            image_path = all_image_paths[0]

        image = Image.open(image_path)
        return self.transform(image)


class VoxelsField(Field[Tensor]):
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, model_name: str) -> Tensor:
        path = self.root / f"{model_name}.binvox"

        with path.open(mode="rb") as f:
            voxels = read_as_3d_array(f)

        return voxels.data.astype(np.float32)


class OccupancyPointsField(Field[tuple[Tensor, Tensor]]):
    def __init__(self, root: Path, num_points: int) -> None:
        self.root = root
        self.num_points = num_points

    def load(self, model_name: str) -> tuple[Tensor, Tensor]:
        path = self.root / f"{model_name}.npz"

        with np.load(path) as f:
            points, points_occ = f["points"], np.unpackbits(f["occupancies"])
            sample_indices = torch.randperm(len(points))[: self.num_points]

            return points[sample_indices], points_occ[sample_indices].astype(np.float32)


class BuildingNetDataset(Dataset):
    """
    Our dataset has the following structure:
    / (root)
    - splits/
    -- dataset_models.txt (1848 model names without extensions)
    -- train_split.txt (1480 model names without extensions)
    -- test_split.txt (180 model names without extensions)
    -- val_split.txt (186 model names without extensions)
    - meshes/ (1939 OBJ files and associated materials, filename format {model_name}.obj)
    - images/ (23256 PNG images, filename format {model_name}.obj_{i}.png for i in {0–11})
    - voxels/ (1942 BINVOX files represented 32x32x32 voxel grids of the OBJS from the meshes directory; filename format {model_name}.binvox)

    Our dataset should load the model names from the given split (or all model
    names if given no split) and return an object containing the mesh, the voxels,
    and the associated images for the given index. It should return these things
    in the exact same format that the Shapenet Dataset does.
    """

    manifest: list[str]
    images_field: ImagesField

    def __init__(
        self,
        dataset_root: Path,
        num_sdf_points: int,
        num_pc_points: int,
        image_resolution: int,
        split: Split = Split.ALL,
    ) -> None:
        split_dir = dataset_root / "splits"
        manifest_path = split_dir / split_mapping[split]

        with open(manifest_path) as f:
            self.manifest = f.readlines()

        self.images_field = ImagesField(
            dataset_root / "images", image_resolution, split == Split.TRAIN
        )
        self.voxels_field = VoxelsField(dataset_root / "voxels")
        self.occ_points_field = OccupancyPointsField(
            dataset_root / "occupancy_points", num_sdf_points
        )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index) -> dict:
        model_name = self.manifest[index].strip()
        images = self.images_field.load(model_name)
        voxels = self.voxels_field.load(model_name)
        points, points_occ = self.occ_points_field.load(model_name)

        return asdict(BuildingNetDatum(images, voxels, points, points_occ))
