####### Code build on top of  https://github.com/autonomousvision/occupancy_networks

import glob
import logging
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from .binvox_rw import read_as_3d_array

logger = logging.getLogger(__name__)


def make_3d_grid(bb_min, bb_max, shape):
    """Makes a 3D grid.
    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)
    p = p.detach().numpy()
    return p


def point_cloud_to_volume(points, vsize, radius=1.0):
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    # print(locations)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
    points = points / furthest_distance
    return points


def unit_variance_2(points):
    centroid = points.mean(axis=0).reshape(1, 3)
    scale = points.flatten().std().reshape(1, 1)

    points = (points - centroid) / scale
    return points


def unit_variance(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    points = points / np.std(points, axis=0)
    return points


def volume_to_point_cloud(vol):
    """vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
    return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert vol.shape[1] == vsize and vol.shape[1] == vsize
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


class Field(object):
    """Data fields class."""

    def load(self, split: str, data_path: str, idx: int, category: str):
        """Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        """
        raise NotImplementedError

    def check_complete(self, files):
        """Checks if set is complete.

        Args:
            files: files
        """
        raise NotImplementedError


class VoxelsField(Field):
    """Voxel field class.
    It provides the class used for voxel-based data. Voxel data is stored as 3D numpy array.
    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    """

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, split, model_path, idx, category):
        """Loads the data point.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, "rb") as f:
            voxels = read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        """Check if field is complete.

        Args:
            files: files
        """
        complete = self.file_name in files
        return complete

def to_rgb(image):
    return image.convert("RGB")

class ImagesField(Field):
    """Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        with_camera (bool): whether camera data should be provided
    """

    def __init__(
        self,
        folder_name,
        transform=None,
        extension="jpg",
        with_camera=False,
        n_px=224,
    ):
        self.folder_name = folder_name
        # self.transform = transform
        self.extension = extension
        self.with_camera = with_camera

        self.transform = Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                to_rgb,
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def load(self, split, model_path, idx, category):
        """Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, "*.%s" % self.extension))
        if split == "train":
            # Generate random views when in training
            idx_img = random.randint(0, len(files) - 1)
        else:
            idx_img = 0
        filename = files[idx_img]

        image = Image.open(filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        data = {None: image}

        if self.with_camera:
            camera_file = os.path.join(folder, "cameras.npz")
            camera_dict = np.load(camera_file)
            Rt = camera_dict["world_mat_%d" % idx_img].astype(np.float32)
            K = camera_dict["camera_mat_%d" % idx_img].astype(np.float32)
            data["world_mat"] = Rt
            data["camera_mat"] = K

        return data

    def check_complete(self, files):
        """Check if field is complete.

        Args:
            files: files
        """
        complete = self.folder_name in files
        # TODO: check camera
        return complete


class PointsField(Field):
    """Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    """

    def __init__(
        self, file_name, transform=None, with_transforms=False, unpackbits=False
    ):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, split, model_path, idx, category):
        """Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)

        points = points_dict["points"].astype(np.float32)
        occupancies = points_dict["occupancies"]
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[: points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            "occ": occupancies,
        }

        if self.with_transforms:
            data["loc"] = points_dict["loc"].astype(np.float32)
            data["scale"] = points_dict["scale"].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data


class PointCloudField(Field):
    """Point cloud field.
    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.
    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    """

    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, split, model_path, idx, category):
        """Loads the data point.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict["points"].astype(np.float32)
        normals = pointcloud_dict["normals"].astype(np.float32)

        data = {
            None: points,
            "normals": normals,
        }

        if self.with_transforms:
            data["loc"] = pointcloud_dict["loc"].astype(np.float32)
            data["scale"] = pointcloud_dict["scale"].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        """Check if field is complete.

        Args:
            files: files
        """
        complete = self.file_name in files
        return complete


def offset_points(points, normals, distance_std=0.02):
    distances = np.random.normal(scale=distance_std, size=points.shape[0])
    distances = np.expand_dims(distances, axis=1)
    offsets = normals * distances
    return points + offsets, distances


class Shapes3dDataset(Dataset):
    """3D Shapes dataset class."""

    def __init__(
        self,
        dataset_folder,
        fields,
        split,
        categories=None,
        transform=None,
        num_points=2048,
        num_sdf_points=5000,
        norm=False,
        sampling_type=None,
    ):
        """Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            transform (callable): transformation applied to data points
        """
        # Attributes
        self.split = split
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.transform = transform
        self.num_points = num_points
        self.num_sdf_points = num_sdf_points
        self.norm = norm
        self.sampling_type = sampling_type

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [
                c for c in categories if os.path.isdir(os.path.join(dataset_folder, c))
            ]

        sorted_categories = sorted(categories)
        category_map = {}
        label = 0
        for i in sorted_categories:
            category_map[i] = label
            label = label + 1
        self.category_map = category_map

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, "metadata.yaml")

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {c: {"id": c, "name": "n/a"} for c in categories}
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]["idx"] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning("Category %s does not exist in dataset.", c)

            split_file = os.path.join(subpath, split + ".lst")
            with open(split_file, "r") as f:
                models_c = f.read().split("\n")

            self.models += [{"category": c, "model": m} for m in models_c]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.models)

    def __getitem__(self, idx):
        """Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        """
        category = self.models[idx]["category"]
        model = self.models[idx]["model"]
        c_idx = self.metadata[category]["idx"]

        model_path = os.path.join(self.dataset_folder, category, model)

        data = {}
        for field_name, field in self.fields.items():
            field_data = field.load(self.split, model_path, idx, c_idx)

            if isinstance(field_data, dict):
                # If the field returns a dict, flatten its top-level keys into
                # the form `{field_name}.{key}`.
                for key, value in field_data.items():
                    if key is None:
                        # TODO: When can `key` possibly be None?
                        data[field_name] = value
                    else:
                        data[f"{field_name}.{key}"] = value
            else:
                data[field_name] = field_data

        # Randomly sample `num_points` points from the pointcloud
        total_list = list(range(len(data["pointcloud"])))
        random_index = random.sample(total_list, self.num_points)
        if self.norm:
            data["pc_org"] = unit_variance_2(data["pointcloud"][random_index])
        else:
            data["pc_org"] = data["pointcloud"][random_index]

        data["category"] = category
        data["label"] = self.category_map[category]

        data["idx"] = idx

        # Randomly sample `num_sdf_points` from the implicit points
        total_list = list(range(len(data["points"])))
        random_index_sdf = random.sample(total_list, self.num_sdf_points)
        data["points"] = data["points"][random_index_sdf]
        data["points.occ"] = data["points.occ"][random_index_sdf]

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        """Tests if model is complete.

        Args:
            model (str): modelname
        """
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warning('Field "%s" is incomplete: %s', field_name, model_path)
                return False

        return True
