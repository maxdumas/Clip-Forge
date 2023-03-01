import argparse
import io
import os
from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from dataset import shapenet_dataset
from dataset.datamodule import ShapeNetDataModule
from networks.autoencoder import Autoencoder
from utils.helper import set_seed
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
                    > pl_module.args.threshold
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


def parsing() -> Any:
    parser = argparse.ArgumentParser()

    ### Sub Network details
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
        "--encoder_type",
        type=str,
        default="Voxel_Encoder_BN",
        help="what is the encoder",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="Occ_Simple_Decoder",
        help="what is the decoder",
    )
    parser.add_argument(
        "--emb_dims", type=int, default=128, help="Dimension of embedding"
    )
    parser.add_argument(
        "--last_feature_transform",
        type=str,
        default="add_noise",
        help="add_noise or none",
    )

    ### Dataset details
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

    ### training details
    parser.add_argument("--train_mode", type=str, default="train", help="train or test")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="W&B checkpoint name from which to load the Autoencoder",
    )
    parser.add_argument(
        "--gpus", nargs="+", default=os.environ.get("SM_NUM_GPUS", "0"), help="GPU list"
    )
    parser.add_argument(
        "--optimizer", type=str, choices=("SGD", "Adam"), default="Adam"
    )
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
        "--sampling_type",
        type=str,
        default=None,
        help="what sampling type: None--> Uniform",
    )

    args = parser.parse_args()
    return args


def main():
    args = parsing()
    set_seed(args.seed)

    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(
        project="clip_forge",
        name=os.environ.get("TRAINING_JOB_NAME", "clip-forge-autoencoder"),
        log_model="all",
    )
    wandb_logger.experiment.config.update(args)

    # Loading networks
    if args.checkpoint is not None:
        # If a checkpoint name is explicitly provided, load that checkpoint
        checkpoint = wandb_logger.use_artifact(args.checkpoint, "model")
        checkpoint_dir = checkpoint.download()
        checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
        print(f"Loading specified W&B Checkpoint from {checkpoint_path}.")
        net = Autoencoder.load_from_checkpoint(checkpoint_path)
    elif os.path.exists(os.path.join("/opt/ml/checkpoints", "last.ckpt")):
        # Restore any checkpoint present in /opt/ml/checkpoints, as this
        # represents a checkpoint that was pre-loaded from SageMaker. We need to
        # do this in order to be able to use Spot training, as we need this
        # script to be able to automatically recover after being interrupted.
        checkpoint_path = os.path.join("/opt/ml/checkpoints", "last.ckpt")
        print(
            f"Auto-loading existing SageMaker checkpoint from {checkpoint_path}. Are we resuming after an interruption?"
        )
        net = Autoencoder.load_from_checkpoint(checkpoint_path)
    else:
        net = Autoencoder(args)

    wandb_logger.watch(net)

    datamodule = ShapeNetDataModule(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        fields={
            "pointcloud": shapenet_dataset.PointCloudField("pointcloud.npz"),
            "points": shapenet_dataset.PointsField("points.npz", unpackbits=True),
            "voxels": shapenet_dataset.VoxelsField("model.binvox"),
        },
        categories=args.categories,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_points=args.num_points,
        num_sdf_points=args.num_sdf_points,
        test_num_sdf_points=args.test_num_sdf_points,
        sampling_type=args.sampling_type,
    )

    checkpoint_callback = ModelCheckpoint(
        "/opt/ml/checkpoints",
        monitor="Loss/val",
        mode="max",
        every_n_epochs=5,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="Loss/val", mode="max")
    sampling_callback = LogPredictionSamplesCallback()
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, sampling_callback],
        # accelerator="gpu",
        # devices=args.gpus,
        precision=16,
    )

    if args.train_mode == "test":
        trainer.test(net, datamodule=datamodule)
    else:  # train mode
        trainer.fit(net, datamodule=datamodule)


if __name__ == "__main__":
    main()
