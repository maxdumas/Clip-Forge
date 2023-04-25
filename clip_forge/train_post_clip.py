from pathlib import Path
import argparse
import io
import os

# Hack to fix broken checkpoint imports from old versions of the code structure
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import clip
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from clip.model import CLIP
from PIL import Image
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

from .dataset.datamodule import BuildingNetEmbeddingDataModule
from .networks.autoencoder import Autoencoder
from .networks.latent_flows import LatentFlows
from .utils import helper
from .utils.visualization import make_3d_grid, multiple_plot, multiple_plot_voxel


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
        batch: dict,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Called when the validation batch ends. This renders an image of the
        reconstructed model next to the original model to compare results."""

        num_figs = 3
        voxel_size = 32
        shape = (voxel_size, voxel_size, voxel_size)
        p = make_3d_grid([-0.5] * 3, [+0.5] * 3, shape, device=self.autoencoder.device)
        query_points = p.expand(num_figs, *p.size())

        if batch_idx != 0:
            # We only want to generate prediction images on the first batch of the epoch
            return

        samples = []
        for text_in in self.text_query:
            text = clip.tokenize([text_in]).to(self.autoencoder.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            noise = torch.Tensor(num_figs, pl_module.num_inputs).normal_()
            decoder_embs = pl_module.sample(
                num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs, 1)
            )

            out = self.autoencoder.decoding(decoder_embs, query_points)

            if self.output_type == "Implicit":
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
            elif self.output_type == "Pointcloud":
                pred = out.detach().cpu().numpy()
                fig = multiple_plot(pred)

            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            im = Image.open(buf)
            samples.append(wandb.Image(im, caption=text_in))
            plt.close(fig)

        trainer.logger.experiment.log({"samples": samples})


def get_local_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", None),
        help="Dataset path",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="Shapenet", help="Dataset path"
    )
    parser.add_argument("--num_points", type=int, default=2025, help="Number of points")
    parser.add_argument(
        "--num_sdf_points", type=int, default=5000, help="Number of points"
    )
    parser.add_argument(
        "--test_num_sdf_points", type=int, default=30000, help="Number of points"
    )
    parser.add_argument("--categories", nargs="+", default=None, metavar="N")
    parser.add_argument(
        "--num_blocks", type=int, default=5, help="Num of blocks for prior"
    )
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
        "--optimizer", type=str, choices=("SGD", "Adam"), default="Adam"
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Dimension of embedding"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32, help="Dimension of embedding"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Threshold for voxel stuff"
    )
    parser.add_argument(
        "--noise", type=str, default="add", metavar="N", help="add or remove"
    )
    parser.add_argument(
        "--seed_nf", type=int, default=1, metavar="N", help="add or remove"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_local_parser()

    manual_seed = args.seed_nf
    helper.set_seed(manual_seed)

    wandb_logger = WandbLogger(
        project="clip_forge_latent_flows",
        name=os.environ.get("TRAINING_JOB_NAME", None),
        log_model="all",
    )
    wandb_logger.log_hyperparams(args)

    # Load Autoencoder from checkpoint generated by train_autoencoder.py
    # TODO: Figure out how to do this more elegantly
    checkpoint = wandb_logger.use_artifact(args.autoencoder_checkpoint, "model")
    checkpoint_dir = checkpoint.download()
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")

    # Load Datamodule
    dm = BuildingNetEmbeddingDataModule(
        "cuda",
        checkpoint_path,
        args.clip_model_type,
        args.input_type,
        args.dataset_name,
        Path(args.dataset_path),
        args.num_views,
        args.batch_size,
        args.test_batch_size,
        args.num_points,
        args.num_sdf_points,
        args.test_num_sdf_points,
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
            dm.cond_emb_dim,
            flow_type=args.flow_type,
            num_blocks=args.num_blocks,
            num_hidden=args.num_hidden,
        )

    wandb_logger.watch(latent_flow_network)

    checkpoint_callback = ModelCheckpoint(
        # If we detect that we're in SageMaker training, use standard SageMaker
        # checkpoint dir. Otherwise use standard PyTorch Lightning checkpoint
        # directory.
        "/opt/ml/checkpoints" if "SM_MODEL_DIR" in os.environ else None,
        monitor="Loss/val",
        mode="max",
        every_n_epochs=10,
    )
    # early_stop_callback = EarlyStopping(monitor="Loss/val", mode="min", patience=15)
    sampling_callback = LogPredictionSamplesCallback(
        args.text_query, args.threshold, args.output_type, dm.autoencoder, dm.clip_model
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, sampling_callback],
        accelerator="auto",
        precision="16-mixed",
        # TODO: make this configurable
        check_val_every_n_epoch=50,
        log_every_n_steps=1,
    )

    trainer.fit(
        latent_flow_network,
        datamodule=dm,
    )


if __name__ == "__main__":
    main()
