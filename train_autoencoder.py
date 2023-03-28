import io
import os
from typing import Any, Optional
from pathlib import Path
import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from torch.utils.data import DataLoader

import wandb
from dataset.buildingnet_dataset import BuildingNetDataset, Split
from networks.autoencoder import Autoencoder
from utils.visualization import multiple_plot_voxel, plot_real_pred
from utils.helper import OutputType


class AutoencoderShapeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Path,
        batch_size: int = 32,
        test_batch_size: int = 32,
        num_points: int = 2025,
        num_sdf_points: int = 5000,
        test_num_sdf_points: int = 5000,
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

    def setup(self, stage: str) -> None:
        base_args = {
            "dataset_root": self.dataset_path,
            "num_sdf_points": self.num_sdf_points,
            "num_pc_points": self.num_points,
            "image_resolution": 224,  # TODO: Avoid hardcoding this
        }
        if stage == "fit":
            self.train_dataset = BuildingNetDataset(**base_args, split=Split.TRAIN)
            self.val_dataset = BuildingNetDataset(**base_args, split=Split.VAL)
        elif stage == "test":
            self.test_dataset = BuildingNetDataset(**base_args, split=Split.TEST)

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
    batch_sample_indices = set(random.choices(list(range(32)), k=4))
    
    def on_validation_batch_end(
        self,
        trainer,
        pl_module: Autoencoder,
        outputs: torch.Tensor,
        batch: dict,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Called when the validation batch ends. This renders an image of the
        reconstructed model next to the original model to compare results."""

        if batch_idx in self.batch_sample_indices:
            # We only want to generate prediction images on the first few batches in the epoch
            return

        def generate_plot_image(outputs):
            match pl_module.output_type:
                case OutputType.IMPLICIT:
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
                case OutputType.POINTCLOUD:
                    gt = batch["pc_org"].type(torch.FloatTensor)

                    fig = plot_real_pred(gt.detach().cpu().numpy(), outputs.numpy(), 1)

            with io.BytesIO() as buf:
                fig.savefig(buf)
                buf.seek(0)
                im = Image.open(buf)
                plt.close(fig)
                return wandb.Image(im)

        fixed_im = generate_plot_image(outputs)
        trainer.logger.experiment.log({f"samples_{batch_idx}": [fixed_im]})


class AutoEncoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("model.test_batch_size", "data.test_batch_size")
        parser.link_arguments("model.num_points", "data.num_points")

        parser.add_argument("--ckpt_path", type=str, default=None)


def main():
    torch.set_float32_matmul_precision("medium")

    checkpoint_callback = ModelCheckpoint(
        # TODO: Implement a better check that we are in SageMaker
        "/opt/ml/checkpoints" if os.path.exists("/opt/ml") else None,
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
    wandb_logger.log_hyperparams(cli.config)

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
