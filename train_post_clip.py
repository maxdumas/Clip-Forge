import argparse
import io
import logging
import os
from typing import Any

import clip
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from clip.model import CLIP
from PIL import Image
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset import shapenet_dataset
from networks.autoencoder import Autoencoder
from networks.latent_flows import LatentFlows
from utils import helper
from utils.visualization import make_3d_grid, multiple_plot, multiple_plot_voxel


def experiment_name2(args):
    tokens = [
        "Clip_Conditioned",
        args.flow_type,
        args.num_blocks,
        args.checkpoint,
        args.num_views,
        args.clip_model_type,
        args.num_hidden,
        args.seed_nf,
    ]

    if args.noise != "add":
        tokens.append("no_noise")

    return "_".join(map(str, tokens))


def get_clip_model(args) -> tuple[Any, CLIP]:
    if args.clip_model_type == "B-16":
        print("Bigger model is being used B-16")
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=args.device)
        cond_emb_dim = 512
    elif args.clip_model_type == "RN50x16":
        print("Using the RN50x16 model")
        clip_model, clip_preprocess = clip.load("RN50x16", device=args.device)
        cond_emb_dim = 768
    else:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
        cond_emb_dim = 512

    input_resolution = clip_model.visual.input_resolution
    vocab_size = clip_model.vocab_size
    print("cond_emb_dim:", cond_emb_dim)
    print("Input resolution:", input_resolution)
    print("Vocab size:", vocab_size)
    args.n_px = input_resolution
    args.cond_emb_dim = cond_emb_dim
    return args, clip_model


def get_dataloader(args, split="train", dataset_flag=False):
    dataset_name = args.dataset_name

    if dataset_name == "Shapenet":
        pointcloud_field = shapenet_dataset.PointCloudField("pointcloud.npz")
        points_field = shapenet_dataset.PointsField("points.npz", unpackbits=True)
        voxel_fields = shapenet_dataset.VoxelsField("model.binvox")

        if split == "train":
            image_field = shapenet_dataset.ImagesField(
                "img_choy2016", random_view=True, n_px=args.n_px
            )
        else:
            image_field = shapenet_dataset.ImagesField(
                "img_choy2016", random_view=False, n_px=args.n_px
            )

        fields = {}

        fields["pointcloud"] = pointcloud_field
        fields["points"] = points_field
        fields["voxels"] = voxel_fields
        fields["images"] = image_field

        def my_collate(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)

        if split == "train":
            dataset = shapenet_dataset.Shapes3dDataset(
                args.dataset_path,
                fields,
                split=split,
                categories=args.categories,
                no_except=True,
                transform=None,
                num_points=args.num_points,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True,
                collate_fn=my_collate,
            )
            total_shapes = len(dataset)
        else:
            dataset = shapenet_dataset.Shapes3dDataset(
                args.dataset_path,
                fields,
                split=split,
                categories=args.categories,
                no_except=True,
                transform=None,
                num_points=args.num_points,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.test_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=my_collate,
            )
            total_shapes = len(dataset)

        if dataset_flag == True:
            return dataloader, total_shapes, dataset

        return dataloader, total_shapes

    else:
        raise ValueError("Dataset name is not defined {}".format(dataset_name))


def get_condition_embeddings(
    args,
    autoencoder: Autoencoder,
    clip_model: CLIP,
    dataloader: DataLoader,
    n_embeddings_per_datum=5,
):
    """
    Given an Autoencoder and CLIP model, generates 3D shape embeddings using the
    Autoencoder and 2D image rendering embeddings using CLIP for every data
    point in the dataset represented by dataloader.

    Note that this implies the dataset must have available 2D image renderings
    of the 3D model.

    Embeddings will be repeatedly generated `n_embeddings_per_datum` times.
    """
    autoencoder.eval()
    clip_model.eval()
    shape_embeddings = []
    cond_embeddings = []
    with torch.no_grad():
        for i in range(0, n_embeddings_per_datum):
            for data in tqdm(dataloader):
                image = data["images"].type(torch.FloatTensor).to(args.device)

                if args.input_type == "Voxel":
                    data_input = data["voxels"].type(torch.FloatTensor).to(args.device)
                elif args.input_type == "Pointcloud":
                    data_input = (
                        data["pc_org"]
                        .type(torch.FloatTensor)
                        .to(args.device)
                        .transpose(-1, 1)
                    )

                shape_emb = autoencoder.encoder(data_input)

                image_features = clip_model.encode_image(image)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                shape_embeddings.append(shape_emb.detach().cpu().numpy())
                cond_embeddings.append(image_features.detach().cpu().numpy())
                # break
            logging.info("Number of views done: %s/%s", i, n_embeddings_per_datum)

        shape_embeddings = np.concatenate(shape_embeddings)
        cond_embeddings = np.concatenate(cond_embeddings)

    logging.info(
        "Embedding Shape %s, Train Condition Embedding %s",
        shape_embeddings.shape,
        cond_embeddings.shape,
    )

    return torch.utils.data.TensorDataset(
        torch.from_numpy(shape_embeddings),
        torch.from_numpy(cond_embeddings),
    )


class LogPredictionSamplesCallback(Callback):
    def __init__(
        self,
        text_query: list[str],
        threshold: float,
        output_type: str,
        autoencoder: Autoencoder,
        clip_model: CLIP,
    ) -> None:
        super().__init__()
        self.text_query = text_query
        self.threshold = threshold
        self.output_type = output_type
        self.autoencoder = autoencoder
        self.clip_model = clip_model

    def on_validation_batch_end(
        self,
        trainer,
        pl_module: LatentFlows,
        outputs: torch.Tensor,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Called when the validation batch ends. This renders an image of the
        reconstructed model next to the original model to compare results."""

        num_figs = 3
        voxel_size = 32
        shape = (voxel_size, voxel_size, voxel_size)
        p = make_3d_grid([-0.5] * 3, [+0.5] * 3, shape)
        query_points = p.expand(num_figs, *p.size())

        if batch_idx != 0:
            # We only want to generate prediction images on the first batch of the epoch
            return

        for text_in in self.text_query:
            text = clip.tokenize([text_in])
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            noise = torch.Tensor(num_figs, pl_module.num_inputs).normal_()
            decoder_embs = pl_module.sample(
                num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs, 1)
            )

            out = self.autoencoder.decoding(decoder_embs, query_points)

            if pl_module.output_type == "Implicit":
                voxels_out = (
                    (
                        out.view(num_figs, voxel_size, voxel_size, voxel_size)
                        > self.threshold
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                fig = multiple_plot_voxel(voxels_out)
            elif pl_module.output_type == "Pointcloud":
                pred = out.detach().cpu().numpy()
                fig = multiple_plot(pred)

            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            im = Image.open(buf)
            trainer.logger.experiment.log({"samples": [wandb.Image(im)]})
            plt.close(fig)


def get_local_parser(mode="args"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_blocks", type=int, default=5, help="Num of blocks for prior"
    )
    parser.add_argument(
        "--flow_type",
        type=str,
        default="realnvp_half",
        choices=["realnvp", "realnvp_half"],
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=1024,
        help="Number of parameter for flow model",
    )
    parser.add_argument(
        "--emb_dims", type=int, default=128, help="Dimension of embedding"
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        help="W&B checkpoint name from which to load the Autoencoder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load latent flow model",
    )
    parser.add_argument(
        "--text_query", nargs="+", default=None, metavar="N", help="text query array"
    )
    parser.add_argument(
        "--num_views", type=int, default=5, metavar="N", help="Number of views"
    )
    parser.add_argument(
        "--clip_model_type",
        type=str,
        default="B-32",
        metavar="N",
        help="what model to use",
    )
    parser.add_argument(
        "--noise", type=str, default="add", metavar="N", help="add or remove"
    )
    parser.add_argument(
        "--seed_nf", type=int, default=1, metavar="N", help="add or remove"
    )
    parser.add_argument(
        "--images_type", type=str, default=None, help="img_choy13 or img_custom"
    )
    parser.add_argument("--n_px", type=int, default=224, help="Resolution of the image")

    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser


def main():
    args = get_local_parser()

    manual_seed = args.seed_nf
    helper.set_seed(manual_seed)

    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(
        project="clip_forge",
        name=os.environ.get("TRAINING_JOB_NAME", "clip-forge-autoencoder"),
        log_model="all",
    )
    wandb_logger.experiment.config.update(args)

    # Loading datasets
    train_dataloader, total_shapes = get_dataloader(args, split="train")
    args.total_shapes = total_shapes
    logging.info("Train Dataset size: %s", total_shapes)
    val_dataloader, total_shapes_val = get_dataloader(args, split="val")
    logging.info("Test Dataset size: %s", total_shapes_val)

    # Loading networks
    # Load CLIP
    args, clip_model = get_clip_model(args)

    # Load Autoencoder from checkpoint generated by train_autoencoder.py
    # TODO: Modularize this and share code with train_autoencoder.py
    checkpoint = wandb_logger.use_artifact(args.autoencoder_checkpoint, "model")
    checkpoint_dir = checkpoint.download()
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    print(f"Loading specified W&B autoencoder Checkpoint from {checkpoint_path}.")
    net = Autoencoder.load_from_checkpoint(checkpoint_path)

    # Load latent flow network
    if args.checkpoint is not None:
        # If a checkpoint name is explicitly provided, load that checkpoint
        checkpoint = wandb_logger.use_artifact(args.checkpoint, "model")
        checkpoint_dir = checkpoint.download()
        checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
        print(f"Loading specified W&B latent flows Checkpoint from {checkpoint_path}.")
        latent_flow_network = LatentFlows.load_from_checkpoint(checkpoint_path)
    elif os.path.exists(os.path.join("/opt/ml/checkpoints", "last.ckpt")):
        # Restore any checkpoint present in /opt/ml/checkpoints, as this
        # represents a checkpoint that was pre-loaded from SageMaker. We need to
        # do this in order to be able to use Spot training, as we need this
        # script to be able to automatically recover after being interrupted.
        checkpoint_path = os.path.join("/opt/ml/checkpoints", "last.ckpt")
        print(
            f"Auto-loading existing SageMaker checkpoint from {checkpoint_path}. Are we resuming after an interruption?"
        )
        latent_flow_network = LatentFlows.load_from_checkpoint(checkpoint_path)
    else:
        latent_flow_network = LatentFlows(
            args.emb_dims,
            args.cond_emb_dim,
            flow_type=args.flow_type,
            num_blocks=args.num_blocks,
            num_hidden=args.num_hidden,
        )

    # Generate TensorDatasets for Autoencoded shape embeddings and CLIP image embeddings
    logging.info("Getting train shape embeddings and condition embedding")
    train_dataset_new = get_condition_embeddings(
        args, net, clip_model, train_dataloader, n_embeddings_per_datum=args.num_views
    )
    train_dataloader_new = DataLoader(
        train_dataset_new,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    logging.info("Getting val shape embeddings and condition embedding")
    val_dataset_new = get_condition_embeddings(
        args, net, clip_model, val_dataloader, n_embeddings_per_datum=1
    )
    val_dataloader_new = DataLoader(
        val_dataset_new,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    checkpoint_callback = ModelCheckpoint(
        "/opt/ml/checkpoints",
        monitor="Loss/val",
        mode="max",
        every_n_epochs=5,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="Loss/val", mode="max")
    sampling_callback = LogPredictionSamplesCallback(
        args.text_query, args.threshold, args.output_type, net, clip_model
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, sampling_callback],
        accelerator="gpu",
        devices=args.gpus,
        precision=16,
    )

    trainer.fit(
        latent_flow_network,
        train_dataloaders=train_dataloader_new,
        val_dataloaders=val_dataloader_new,
    )


if __name__ == "__main__":
    main()
