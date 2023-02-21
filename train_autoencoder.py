import argparse
import io
import os
from typing import Any

import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

import wandb
from dataset import shapenet_dataset
from networks.autoencoder import Autoencoder
from utils import helper
from utils.visualization import multiple_plot_voxel, plot_real_pred


def experiment_name(args):
    tokens = [
        "Autoencoder",
        args.dataset_name,
        args.input_type,
        args.output_type,
        args.emb_dims,
        args.last_feature_transform,
    ]

    if args.categories is not None:
        for i in args.categories:
            tokens.append(i)

    if args.num_sdf_points != 5000:
        tokens.append(args.num_sdf_points)

    tokens.append(args.seed)
    return "_".join(map(str, tokens))


class AutoencoderShapeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        categories: list[str] | None,
        batch_size: int = 32,
        test_batch_size: int = 32,
        num_points: int = 2025,
        num_sdf_points: int = 5000,
        test_num_sdf_points: int = 5000,
        sampling_type: str | None = None,
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
                sampling_type=self.sampling_type,
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
                sampling_type=self.sampling_type,
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
                sampling_type=self.sampling_type,
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
                outputs[0].view(voxel_size, voxel_size, voxel_size)
                > pl_module.args.threshold
            ).numpy()
            real = voxel_32[0].numpy()
            fig = multiple_plot_voxel([real, voxels_out])
        elif pl_module.output_type == "Pointcloud":
            gt = batch["pc_org"].type(torch.FloatTensor)

            fig = plot_real_pred(gt.detach().cpu().numpy(), outputs.numpy(), 1)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        im = Image.open(buf)
        # wandb_logger.log_image(key="samples", images=[im])
        trainer.logger.experiment.log({
            "samples": [wandb.Image(im)]
        })


def parsing(mode="args") -> Any:
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
    parser.add_argument(
        "--reconstruct_loss_type",
        type=str,
        default="sum",
        help="bce or sum (mse) or mean (mse)",
    )
    parser.add_argument(
        "--pc_dims", type=int, default=1024, help="Dimension of embedding"
    )

    ### Dataset details
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
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
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    ### training details
    parser.add_argument("--train_mode", type=str, default="train", help="train or test")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to load"
    )
    parser.add_argument(
        "--use_timestamp",
        action="store_true",
        help="Whether to use timestamp in dump files",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=300000,
        help="How long the training should go on",
    )
    parser.add_argument("--gpu", nargs="+", default="0", help="GPU list")
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

    ### Logging details
    parser.add_argument(
        "--print_every", type=int, default=50, help="Printing the loss every"
    )
    parser.add_argument(
        "--save_every", type=int, default=50, help="Saving the model every"
    )
    parser.add_argument(
        "--validation_every", type=int, default=5000, help="validation set every"
    )
    parser.add_argument(
        "--visualization_every",
        type=int,
        default=10,
        help="visualization of the results every",
    )
    parser.add_argument(
        "--log-level", type=str, choices=("info", "warn", "error"), default="info"
    )
    parser.add_argument(
        "--experiment_type", type=str, default="max", help="experiment type"
    )
    parser.add_argument(
        "--experiment_every", type=int, default=5, help="experiment every "
    )

    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser


def main():
    args = parsing()
    exp_name = experiment_name(args)

    helper.set_seed(args.seed)

    wandb_logger = WandbLogger(project="clip_forge", name=exp_name, log_model="all")
    wandb_logger.experiment.config.update(args)

    # Loading networks
    if args.checkpoint is not None:
        net = Autoencoder.load_from_checkpoint(args.checkpoint)
    else:
        net = Autoencoder(args)

    datamodule = AutoencoderShapeNetDataModule(
        args.dataset_name,
        args.dataset_path,
        args.categories,
        args.batch_size,
        args.test_batch_size,
        args.num_points,
        args.num_sdf_points,
        args.test_num_sdf_points,
        args.sampling_type,
    )

    checkpoint_callback = ModelCheckpoint(monitor="Loss/val", mode="min", every_n_epochs=5, save_last=True)
    sampling_callback = LogPredictionSamplesCallback()
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, sampling_callback])

    if args.train_mode == "test":
        trainer.test(net, datamodule=datamodule)
    else:  # train mode
        trainer.fit(net, datamodule=datamodule)


if __name__ == "__main__":
    main()
