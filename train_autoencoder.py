import io
import os
from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

import wandb
from dataset.datamodule import ShapeNetDataModule
from networks.autoencoder import Autoencoder
from utils.visualization import multiple_plot_voxel, plot_real_pred


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
        ShapeNetDataModule,
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
