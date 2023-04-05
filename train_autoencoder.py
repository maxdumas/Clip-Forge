import io
import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

import wandb
from networks.autoencoder import Autoencoder
from utils.visualization import multiple_plot_voxel, plot_real_pred
from utils.helper import InputType, OutputType
from dataset.datamodule import BuildingNetDataModule


class LogPredictionSamplesCallback(Callback):
    # TODO: Intelligently determine this based on the batch size and total data set size
    # Which batche indices to sample predictions from
    batch_sample_indices = {0, 1}
    # batch_sample_indices = set(random.sample(range(6), k=4))
    # Within a sampled batch, which examples to use to sample predictions
    indices_within_batch = random.sample(range(32), k=2)

    def generate_plot_image(self, pl_module: Autoencoder, outputs, batch: dict, i: int):
        if pl_module.output_type == OutputType.IMPLICIT:
            voxel_32 = batch["voxels"].type(torch.FloatTensor)
            voxel_size = 32

            voxels_out = (
                (
                    outputs[i].view(voxel_size, voxel_size, voxel_size)
                    > pl_module.threshold
                )
                .cpu()
                .numpy()
            )
            real = voxel_32[i].cpu().numpy()
            fig = multiple_plot_voxel([real, voxels_out])
        elif pl_module.output_type == OutputType.POINTCLOUD:
            gt = batch["pc_org"].type(torch.FloatTensor)

            fig = plot_real_pred(gt.detach().cpu().numpy(), outputs.numpy(), 1)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        im = Image.open(buf)
        plt.close(fig)
        return wandb.Image(im)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Autoencoder,
        outputs,  # This is just the training loss
        batch: dict,
        batch_idx: int,
    ) -> None:
        # Only sample training images every 10 epochs and for a random selection of batches in the epoch.
        if (
            trainer.current_epoch % 10 != 0
            or batch_idx not in self.batch_sample_indices
        ):
            return

        # TODO: Implement POINTCLOUD support here.
        assert (
            pl_module.input_type == InputType.VOXELS
            and pl_module.output_type == OutputType.IMPLICIT
        )

        data_input = batch["voxels"]
        query_points = pl_module.create_sampling_grid(data_input)

        # Run prediction (being sure not to affect training)
        with torch.no_grad():
            pl_module.eval()
            outputs, _ = pl_module.forward(data_input, query_points)
            pl_module.train()

        trainer.logger.experiment.log(
            {
                f"samples/train_{batch_idx}_{i}": [
                    self.generate_plot_image(pl_module, outputs, batch, i)
                ]
                for i in self.indices_within_batch
            },
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Autoencoder,
        outputs: torch.Tensor,
        batch: dict,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Called when the validation batch ends. This renders an image of the
        reconstructed model next to the original model to compare results."""

        # Only sample validation images for a random selection of batches in the epoch.
        if batch_idx not in self.batch_sample_indices:
            return

        trainer.logger.experiment.log(
            {
                f"samples/val_{batch_idx}_{i}": [
                    self.generate_plot_image(pl_module, outputs, batch, i)
                ]
                for i in self.indices_within_batch
            },
        )


class AutoEncoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("model.test_batch_size", "data.test_batch_size")
        parser.link_arguments("model.num_points", "data.num_points")

        # parser.add_argument("--ckpt_path", type=str, default=None)

    def search_for_checkpoints(self, wandb_logger: WandbLogger):
        ckpt_path = self.config_init[self.subcommand].ckpt_path
        sm_ckpt_path = os.path.join("/opt/ml/checkpoints", "last.ckpt")
        if ckpt_path is not None:
            # If a checkpoint name is explicitly provided, load that checkpoint from W&B
            checkpoint = wandb_logger.use_artifact(ckpt_path, "model")
            checkpoint_dir = checkpoint.download()
            ckpt_path = os.path.join(checkpoint_dir, "model.ckpt")
            print(f"Loading specified W&B Checkpoint from {ckpt_path}.")
        elif os.path.exists(sm_ckpt_path):
            # Restore any checkpoint present in /opt/ml/checkpoints, as this
            # represents a checkpoint that was pre-loaded from SageMaker. We need to
            # do this in order to be able to use Spot training, as we need this
            # script to be able to automatically recover after being interrupted.
            ckpt_path = sm_ckpt_path
            print(
                f"Auto-loading existing SageMaker checkpoint from {ckpt_path}. Are we resuming after an interruption?"
            )
        self.config_init[self.subcommand].ckpt_path = ckpt_path

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        wandb_logger = WandbLogger(
            project="clip_forge_autoencoder",
            name=os.environ.get("TRAINING_JOB_NAME", None),
            log_model="all",
        )

        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()
        self.trainer.logger = wandb_logger
        wandb_logger.log_hyperparams(self.config)

        self.search_for_checkpoints(wandb_logger)

        self.model = self._get(self.config_init, "model")
        # self.model = torch.compile(self.model)
        wandb_logger.watch(self.model)


def main():
    checkpoint_callback = ModelCheckpoint(
        # TODO: Implement a better check that we are in SageMaker
        "/opt/ml/checkpoints" if os.path.exists("/opt/ml") else None,
        monitor="loss/train/reconstruction",
        mode="min",
        every_n_epochs=10,
    )
    # early_stop_callback = EarlyStopping(
    #    monitor="loss/train/reconstruction",
    #    mode="min",
    #    patience=50,
    #    stopping_threshold=0.0,
    #    # divergence_threshold=0.01,
    #    verbose=True,
    # )
    sampling_callback = LogPredictionSamplesCallback()
    model_summary_callback = ModelSummary(max_depth=-1)

    AutoEncoderCLI(
        Autoencoder,
        BuildingNetDataModule,
        trainer_defaults={
            "callbacks": [
                model_summary_callback,
                checkpoint_callback,
                sampling_callback,
            ]
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
