import os
from typing import Optional
from functools import partial

import clip
import datasets
import torch
from clip.model import CLIP
from torchdata.datapipes.iter import IterableWrapper
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.reading_service import MultiProcessingReadingService

from dataset import shapenet_dataset
from networks.autoencoder import Autoencoder

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


@torch.no_grad()
def get_cond_embeddings(
    device, autoencoder: Autoencoder, clip_model: CLIP, data: dict[str, torch.Tensor]
):
    image = data["images"].type(torch.FloatTensor).to(device)

    if autoencoder.input_type == "Voxel":
        data_input = data["voxels"].type(torch.FloatTensor).to(device)
    elif autoencoder.input_type == "Pointcloud":
        data_input = data["pc_org"].type(torch.FloatTensor).to(device).transpose(-1, 1)

    shape_emb = autoencoder.encoder(data_input)

    image_features = clip_model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return {
        "train_embs": shape_emb,
        "train_cond_embs": image_features,
    }

def get_dataloader(
    dataset_name: str,
    dataset_path: str,
    batch_size: int,
    num_points: int,
    n_px: int,
    autoencoder: Autoencoder,
    clip_model: CLIP,
    split: str,
    device,
    categories: Optional[list[str]] = None,
    n_embeddings_per_datum=5,
):
    if dataset_name != "Shapenet":
        raise ValueError(f"Dataset name is not defined: {dataset_name}.")

    shuffle = drop_last = random_view = split == "train"

    fields = {
        "pointcloud": shapenet_dataset.PointCloudField("pointcloud.npz"),
        "points": shapenet_dataset.PointsField("points.npz", unpackbits=True),
        "voxels": shapenet_dataset.VoxelsField("model.binvox"),
        "images": shapenet_dataset.ImagesField(
            "img_choy2016", random_view=random_view, n_px=n_px
        ),
    }

    dataset = shapenet_dataset.Shapes3dDataset(
        dataset_path,
        fields,
        split=split,
        categories=categories,
        num_points=num_points,
    )


    with torch.no_grad():
        dataset = IterableWrapper(dataset)
        if n_embeddings_per_datum > 1:
            dataset = dataset.repeat(n_embeddings_per_datum)
        if shuffle:
            dataset = dataset.shuffle()
        dataset = (
            dataset
            .sharding_filter()
            .batch(batch_size, drop_last)
            .collate()
            .map(partial(get_cond_embeddings, device, autoencoder, clip_model))
        )

    return DataLoader2(
        dataset,
        reading_service=MultiProcessingReadingService(num_workers=os.cpu_count() or 0, multiprocessing_context="spawn")
    )

def generate_embeddings(autoencoder_checkpoint_path: str, batch_size=32, num_points=2025, device="cuda"):
    clip_model, n_px, cond_emb_dim = get_clip_model("B-32", device)
    autoencoder = Autoencoder.load_from_checkpoint(autoencoder_checkpoint_path).to(device)
    autoencoder.eval()
    clip_model.eval()

    train_dataloader_new = get_dataloader(
        "Shapenet",
        "../shapenet",
        batch_size,
        num_points,
        n_px,
        autoencoder,
        clip_model,
        split="train",
        device=device,
        categories=None,
        n_embeddings_per_datum=1,
    )

    val_dataloader_new = get_dataloader(
        "Shapenet",
        "../shapenet",
        batch_size,
        num_points,
        n_px,
        autoencoder,
        clip_model,
        split="val",
        device=device,
        categories=None,
        n_embeddings_per_datum=1,
    )
    ds = datasets.DatasetDict({
        "train": datasets.Dataset.from_generator(train_dataloader_new.__iter__),
        "val": datasets.Dataset.from_generator(val_dataloader_new.__iter__),
    })

    ds.save_to_disk("./hf_dataset")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    checkpoint_path = "artifacts/model-clip-forge-autoencoder-2023-02-26-06-15-12-280-tpdwmj-algo-1:v151/model.ckpt"

    generate_embeddings(checkpoint_path)