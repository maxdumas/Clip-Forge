import argparse
import io
import os

import clip
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from clip.model import CLIP
from PIL import Image
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper

import wandb
from dataset import shapenet_dataset
from networks.autoencoder import Autoencoder
from networks.latent_flows import LatentFlows
from utils.helper import set_seed
from utils.visualization import make_3d_grid, multiple_plot, multiple_plot_voxel


def get_clip_model(clip_model_type: str, device="cuda") -> tuple[CLIP, int, int]:
    if clip_model_type == "B-16":
        print("Bigger model is being used B-16")
        clip_model, _ = clip.load("ViT-B/16", device=device)
        cond_emb_dim = 512
    elif clip_model_type == "RN50x16":
        print("Using the RN50x16 model")
        clip_model, _ = clip.load("RN50x16", device=device)
        cond_emb_dim = 768
    else:
        clip_model, _ = clip.load("ViT-B/32", device=device)
        cond_emb_dim = 512

    input_resolution = clip_model.visual.input_resolution
    return clip_model, input_resolution, cond_emb_dim




class Embed:
    """
    Given an Autoencoder and CLIP model, generates 3D shape embeddings using the
    Autoencoder and 2D image rendering embeddings using CLIP for every data
    point in the dataset represented by dataloader.

    Note that this implies the dataset must have available 2D image renderings
    of the 3D model.

    Embeddings will be repeatedly generated `n_embeddings_per_datum` times.
    """

    def __init__(
        self,
        input_type: str,
        autoencoder: Autoencoder,
        clip_model: CLIP,
        n_embeddings_per_datum: int = 5,
    ):
        self.input_type = input_type
        self.autoencoder = autoencoder
        self.clip_model = clip_model
        self.n_embeddings_per_datum = n_embeddings_per_datum

    @torch.no_grad()
    def embed(self, datum):
        # TODO: The issue here is that the incoming data has not yet been loaded into a Tensor. We need to convert it to a Tesnor
        image = datum["images"]

        if self.input_type == "Voxel":
            data_input = datum["voxels"]
        elif self.input_type == "Pointcloud":
            data_input = datum["pc_org"].transpose(-1, 1)

        shape_emb = self.autoencoder.encoder(data_input)

        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return shape_emb, image_features

    @torch.no_grad()
    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = [self.embed(batch) for _ in range(self.n_embeddings_per_datum)]
        batch = [b for b in batch if b is not None]
        return batch

class LatentFlowsShapeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        categories: list[str],
        batch_size: int,
        test_batch_size: int,
        num_points: int,
        n_px: int,
        collate_fn
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
        self.n_px = n_px
        self.collate_fn = collate_fn

    def setup(self, stage: str) -> None:
        fields = {
            "pointcloud": shapenet_dataset.PointCloudField("pointcloud.npz"),
            "points": shapenet_dataset.PointsField("points.npz", unpackbits=True),
            "voxels": shapenet_dataset.VoxelsField("model.binvox"),
            "images": shapenet_dataset.ImagesField(
                "img_choy2016", random_view=stage == "fit", n_px=self.n_px
            ),
        }

        if stage == "fit":
            self.train_dataset = shapenet_dataset.Shapes3dDataset(
                    self.dataset_path,
                    fields,
                    split="train",
                    categories=self.categories,
                    num_points=self.num_points,
                )
            self.val_dataset = shapenet_dataset.Shapes3dDataset(
                    self.dataset_path,
                    fields,
                    split="val",
                    categories=self.categories,
                    num_points=self.num_points,
                )
        elif stage == "test":
            self.test_dataset = shapenet_dataset.Shapes3dDataset(
                    self.dataset_path,
                    fields,
                    split="test",
                    categories=self.categories,
                    num_points=self.num_points,
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
            shuffle=True,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 0,
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


def get_local_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs")
    parser.add_argument("--categories", nargs="+", default=None, metavar="N")
    parser.add_argument(
        "--input_type",
        type=str,
        default="Voxel",
        help="What is the input representation",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="Implicit",
        help="What is the output representation",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Threshold for voxel stuff"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=5, help="Num of blocks for prior"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", None),
        help="Dataset path",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="Shapenet", help="Dataset path"
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
    parser.add_argument("--num_points", type=int, default=2025, help="Number of points")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Dimension of embedding"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32, help="Dimension of embedding"
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        help="W&B checkpoint name from which to load the Autoencoder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="W&B checkpoint name from which to load the Latent Flows Network",
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
        "--images_type", type=str, default=None, help="img_choy13 or img_custom"
    )
    parser.add_argument(
        "--gpus", nargs="+", default=os.environ.get("SM_NUM_GPUS", "0"), help="GPU list"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_local_parser()

    set_seed(args.seed)

    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(
        project="clip_forge",
        name=os.environ.get("TRAINING_JOB_NAME", "clip-forge-latent-flows"),
        log_model="all",
    )
    wandb_logger.experiment.config.update(args)

    # Load CLIP
    clip_model, n_px, cond_emb_dim = get_clip_model(args.clip_model_type, device="cpu")

    # Loading networks

    # Load Autoencoder from checkpoint generated by train_autoencoder.py
    # TODO: Modularize this and share code with train_autoencoder.py
    checkpoint = wandb_logger.use_artifact(args.autoencoder_checkpoint, "model")
    checkpoint_dir = checkpoint.download()
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    print(f"Loading specified W&B autoencoder Checkpoint from {checkpoint_path}.")
    net = Autoencoder.load_from_checkpoint(checkpoint_path)

    # Loading datasets
    datamodule = LatentFlowsShapeNetDataModule(
        args.dataset_name,
        args.dataset_path,
        args.categories,
        args.batch_size,
        args.test_batch_size,
        args.num_points,
        n_px,
        collate_fn=Embed(args.input_type, net, clip_model, args.num_views)
    )

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
            cond_emb_dim,
            # lr=args.lr, # TODO: This is coming from argparse as non-null. Unclear why.
            flow_type=args.flow_type,
            num_blocks=args.num_blocks,
            num_hidden=args.num_hidden,
            input_type=args.input_type,
            output_type=args.output_type,
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
        # accelerator="gpu",
        # devices=args.gpus,
        # accelerator="mps",
        # devices="1",
        precision=16,
    )

    trainer.fit(latent_flow_network, datamodule=datamodule)


if __name__ == "__main__":
    main()
