import argparse
import logging
import os.path as osp
from typing import Any

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from dataset import shapenet_dataset
from networks.autoencoder import Autoencoder
from utils import helper, visualization


def experiment_name(args):
    tokens = [
        "Autoencoder",
        args.dataset_name,
        args.input_type,
        args.output_type,
        args.emb_dims,
        args.last_feature_transform,
    ]

    if args.categories != None:
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
        pass

    def train_dataloader(self) -> DataLoader:
        dataset = shapenet_dataset.Shapes3dDataset(
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

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = shapenet_dataset.Shapes3dDataset(
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
        return DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = shapenet_dataset.Shapes3dDataset(
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
        return DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
        )


def visualization_model(model, args, test_dataloader, name_info):
    model.eval()
    test_loader = iter(test_dataloader)
    data = next(test_loader)

    if args.input_type == "Voxel":
        data_input = data["voxels"].type(torch.FloatTensor).to(args.device)
    elif args.input_type == "Pointcloud":
        data_input = (
            data["pc_org"].type(torch.FloatTensor).to(args.device).transpose(-1, 1)
        )

    if args.output_type == "Implicit":
        voxel_32 = data["voxels"].type(torch.FloatTensor).to(args.device)
        voxel_size = 32
        shape = (voxel_size, voxel_size, voxel_size)
        p = 1.1 * visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(
            torch.FloatTensor
        ).to(args.device)
        query_points = p.expand(args.test_batch_size, *p.size())
    elif args.output_type == "Pointcloud":
        query_points = None
        gt = data["pc_org"].type(torch.FloatTensor).to(args.device)

    with torch.no_grad():
        pred, decoder_embs = model(data_input, query_points)

        if name_info is not None:
            save_loc = args.vis_dir + "/" + str(name_info) + "_"
        else:
            save_loc = args.vis_dir + "/"

        if args.output_type == "Implicit":
            voxels_out = (
                (pred[0].view(voxel_size, voxel_size, voxel_size) > args.threshold)
                .detach()
                .cpu()
                .numpy()
            )
            real = voxel_32[0].detach().cpu().numpy()
            visualization.multiple_plot_voxel(
                [real, voxels_out], save_loc=save_loc + "real_pred.png"
            )
            # visualization.save_mesh(voxels_out, out_file=save_loc + "pred.obj")
        elif args.output_type == "Pointcloud":
            visualization.plot_real_pred(
                gt.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                1,
                save_loc=save_loc + "real_pred.png",
            )


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

    manual_seed = args.seed
    helper.set_seed(manual_seed)

    # Create directories for checkpoints and logging
    args.experiment_dir = osp.join("exps", exp_name)
    args.checkpoint_dir = osp.join(args.experiment_dir, "checkpoints")
    args.vis_dir = osp.join(args.experiment_dir, "vis_dir") + "/"
    args.generate_dir = osp.join(args.experiment_dir, "generate_dir") + "/"

    if args.train_mode == "test":
        test_log_filename = osp.join(args.experiment_dir, "test_log.txt")
        helper.setup_logging(test_log_filename, args.log_level, "w")
        args.examplar_generate_dir = (
            osp.join(args.experiment_dir, "exam_generate_dir") + "/"
        )
        helper.create_dir(args.examplar_generate_dir)
        # The directory where generated images will be stored
        args.vis_gen_dir = osp.join(args.experiment_dir, "vis_gen_dir") + "/"
        helper.create_dir(args.vis_gen_dir)
    else:
        log_filename = osp.join("exps", exp_name, "log.txt")
        helper.create_dir(args.experiment_dir)
        helper.create_dir(args.checkpoint_dir)
        helper.create_dir(args.vis_dir)
        helper.create_dir(args.generate_dir)
        helper.setup_logging(log_filename, args.log_level, "w")

    logging.info("Experiment name: %s", exp_name)
    logging.info("%s", args)

    wandb = WandbLogger(name="clip_forge/autoencoder", log_model=True)
    wandb.experiment.config.update(args)

    # Loading networks
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint_dir + "/" + args.checkpoint + ".pt")
        net = Autoencoder.load_from_checkpoint(checkpoint)
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

    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="Loss/val")
    trainer = pl.Trainer(logger=wandb, callbacks=[checkpoint_callback])

    if args.train_mode == "test":
        trainer.test(net, datamodule=datamodule)

    else:  # train mode
        trainer.fit(net, datamodule=datamodule)


if __name__ == "__main__":
    main()
