import io
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from torch.utils.data import DataLoader

import wandb
from dataset import shapenet_dataset
from networks.autoencoder import Autoencoder
from utils.visualization import multiple_plot_voxel, plot_real_pred


class AutoencoderShapeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        categories: Optional[list[str]],
        batch_size: int = 32,
        test_batch_size: int = 32,
        num_points: int = 2025,
        num_sdf_points: int = 5000,
        test_num_sdf_points: int = 5000,
        sampling_type: Optional[str] = None,
    ):
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

        self.fields = {
            "pointcloud": shapenet_dataset.PointCloudField("pointcloud.npz"),
            "points": shapenet_dataset.PointsField("points.npz", unpackbits=True),
            "voxels": shapenet_dataset.VoxelsField("model.binvox"),
        }

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = shapenet_dataset.Shapes3dDataset(
                self.dataset_path,
                self.fields,
                split="train",
                categories=self.categories,
                no_except=True,
                transform=None,
                num_points=self.num_points,
                num_sdf_points=self.num_sdf_points,
            )
            self.val_dataset = shapenet_dataset.Shapes3dDataset(
                self.dataset_path,
                self.fields,
                split="val",
                categories=self.categories,
                no_except=True,
                transform=None,
                num_points=self.num_points,
                num_sdf_points=self.test_num_sdf_points,
            )
        elif stage == "test":
            self.test_dataset = shapenet_dataset.Shapes3dDataset(
                self.dataset_path,
                self.fields,
                split="test",
                categories=self.categories,
                no_except=True,
                transform=None,
                num_points=self.num_points,
                num_sdf_points=self.test_num_sdf_points,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=os.cpu_count() or 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
        )


class LogPredictionSamplesCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer,
        pl_module: Autoencoder,
        outputs: torch.Tensor,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Called when the validation batch ends. This renders an image of the
        reconstructed model next to the original model to compare results."""

        if batch_idx != 0:
            # We only want to generate prediction images on the first batch of the epoch
            return

        if pl_module.output_type == "Implicit":
            voxel_32 = batch["voxels"].type(torch.FloatTensor)
            voxel_size = 32

            voxels_out = (
                (
                    outputs[0].view(voxel_size, voxel_size, voxel_size)
                    > pl_module.threshold
                )
                .cpu()
                .numpy()
            )
            real = voxel_32[0].cpu().numpy()
            fig = multiple_plot_voxel([real, voxels_out])
        elif pl_module.output_type == "Pointcloud":
            gt = batch["pc_org"].type(torch.FloatTensor)

            fig = plot_real_pred(gt.detach().cpu().numpy(), outputs.numpy(), 1)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        im = Image.open(buf)
        trainer.logger.experiment.log({"samples": [wandb.Image(im)]})
        plt.close(fig)


class AutoEncoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("model.test_batch_size", "data.test_batch_size")
        parser.link_arguments("model.num_points", "data.num_points")

        parser.add_argument("--ckpt_path", type=str, default=None)


def main():
    torch.set_float32_matmul_precision("medium")

    checkpoint_callback = ModelCheckpoint(
        "/opt/ml/checkpoints",
        monitor="Loss/val",
        mode="max",
        every_n_epochs=5,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="Loss/val", mode="max")
    sampling_callback = LogPredictionSamplesCallback()
    cli = AutoEncoderCLI(
        Autoencoder,
        AutoencoderShapeNetDataModule,
        trainer_defaults={
            "callbacks": [checkpoint_callback, early_stop_callback, sampling_callback]
        },
        run=False,
        save_config_callback=None,
    )

    wandb_logger = WandbLogger(
        project="clip_forge",
        name=os.environ.get("TRAINING_JOB_NAME", "clip-forge-autoencoder"),
        log_model="all",
    )
    cli.trainer.logger = wandb_logger
    wandb_logger.experiment.config.update(cli.config.as_flat())

    ckpt_path = cli.config.ckpt_path
    if ckpt_path is not None:
        # If a checkpoint name is explicitly provided, load that checkpoint from W&B
        checkpoint = wandb_logger.use_artifact(ckpt_path, "model")
        checkpoint_dir = checkpoint.download()
        ckpt_path = os.path.join(checkpoint_dir, "model.ckpt")
        print(f"Loading specified W&B Checkpoint from {ckpt_path}.")
    elif os.path.exists(os.path.join("/opt/ml/checkpoints", "last.ckpt")):
        # Restore any checkpoint present in /opt/ml/checkpoints, as this
        # represents a checkpoint that was pre-loaded from SageMaker. We need to
        # do this in order to be able to use Spot training, as we need this
        # script to be able to automatically recover after being interrupted.
        ckpt_path = os.path.join("/opt/ml/checkpoints", "last.ckpt")
        print(
            f"Auto-loading existing SageMaker checkpoint from {ckpt_path}. Are we resuming after an interruption?"
        )

    wandb_logger.watch(cli.model)

    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
