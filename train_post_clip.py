import logging
import os.path as osp
from typing import Any

import clip
import numpy as np
import torch
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import shapenet_dataset
from networks import latent_flows
from networks.autoencoder import Autoencoder
from train_autoencoder import experiment_name, parsing
from utils import helper, visualization


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


def generate_on_query_text(args, clip_model, autoencoder, latent_flow_model):
    autoencoder.eval()
    latent_flow_model.eval()
    clip_model.eval()
    save_loc = args.generate_dir + "/"
    num_figs = 3
    with torch.no_grad():
        voxel_size = 32
        shape = (voxel_size, voxel_size, voxel_size)
        p = (
            visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape)
            .type(torch.FloatTensor)
            .to(args.device)
        )
        query_points = p.expand(num_figs, *p.size())

        for text_in in args.text_query:
            text = clip.tokenize([text_in]).to(args.device)
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            noise = torch.Tensor(num_figs, args.emb_dims).normal_().to(args.device)
            decoder_embs = latent_flow_model.sample(
                num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs, 1)
            )

            out = autoencoder.decoding(decoder_embs, query_points)

            if args.output_type == "Implicit":
                voxels_out = (
                    (
                        out.view(num_figs, voxel_size, voxel_size, voxel_size)
                        > args.threshold
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                visualization.multiple_plot_voxel(
                    voxels_out, save_loc=save_loc + "{}_text_query.png".format(text_in)
                )
            elif args.output_type == "Pointcloud":
                pred = out.detach().cpu().numpy()
                visualization.multiple_plot(
                    pred, save_loc=save_loc + "{}_text_query.png".format(text_in)
                )

    latent_flow_model.train()


def train_one_epoch(
    args: Any,
    latent_flow_model: latent_flows.FlowSequential,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Adam,
    epoch: int,
    loss_meter: helper.AverageMeter,
    writer: SummaryWriter
):
    latent_flow_model.train()
    iteration = 0

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            osp.join(args.experiment_dir, "tb_logs")
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for data in train_dataloader:
            iteration = iteration + 1
            optimizer.zero_grad()

            # Load appropriate input data from the training set
            train_embs, train_cond_embs = data
            train_embs = train_embs.type(torch.FloatTensor).to(args.device)
            train_cond_embs = train_cond_embs.type(torch.FloatTensor).to(args.device)

            # Add noise to improve robustness
            if args.noise == "add":
                train_embs = train_embs + 0.1 * torch.randn(
                    train_embs.size(0), args.emb_dims
                ).to(args.device)

            # Compute loss (prediction occurs within this method)
            loss_log_prob = -latent_flow_model.log_prob(
                train_embs, train_cond_embs
            ).mean()
            loss = loss_log_prob

            # Compute loss gradient
            loss.backward()

            # Perform a single optimizer step, updating model parameters according
            # to gradient
            optimizer.step()

            # Track metrics
            writer.add_scalar("Loss/train", loss, epoch * args.num_iterations + iteration)
            prof.step()
            loss_meter.update(loss)

            if iteration % args.print_every == 0:
                logging.info(
                    "[Train] Epoch %s, Iteration %s loss: %s ",
                    epoch,
                    iteration,
                    loss_meter.avg
                )


def val_one_epoch(args, latent_flow_model, val_dataloader, epoch, writer):
    loss_array = []
    latent_flow_model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            train_embs, train_cond_embs = data
            train_embs = train_embs.type(torch.FloatTensor).to(args.device)
            train_cond_embs = train_cond_embs.type(torch.FloatTensor).to(args.device)
            loss = -latent_flow_model.log_prob(
                train_embs, train_cond_embs
            ).mean()

            loss_array.append(loss.item())
    mean_loss = np.mean(np.asarray(loss_array))
    writer.add_scalar("Loss/val", mean_loss, epoch * args.num_iterations)
    logging.info(
        "[VAL] Epoch %s Train loss %s",
        epoch,
        mean_loss,
    )
    return mean_loss


def get_local_parser(mode="args"):
    parser = parsing(mode="parser")
    parser.add_argument(
        "--num_blocks", type=int, default=5, help="Num of blocks for prior"
    )
    parser.add_argument(
        "--flow_type",
        type=str,
        default="realnvp_half",
        help="flow type: mf, glow, realnvp ",
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=1024,
        help="Number of parameter for flow model",
    )
    parser.add_argument(
        "--latent_load_checkpoint",
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
    exp_name = experiment_name(args)
    exp_name_2 = experiment_name2(args)

    manual_seed = args.seed_nf
    helper.set_seed(manual_seed)

    # Create directories for checkpoints and logging
    args.experiment_dir = osp.join("exps", exp_name, exp_name_2)
    args.experiment_dir_base = osp.join("exps", exp_name)
    args.checkpoint_dir = osp.join(args.experiment_dir, "checkpoints")
    args.checkpoint_dir_base = osp.join(args.experiment_dir_base, "checkpoints")
    args.vis_dir = osp.join(args.experiment_dir, "vis_dir") + "/"
    args.generate_dir = osp.join(args.experiment_dir, "generate_dir") + "/"

    log_filename = osp.join(args.experiment_dir, "log.txt")

    helper.create_dir(args.checkpoint_dir)
    helper.create_dir(args.vis_dir)
    helper.create_dir(args.generate_dir)

    if args.train_mode == "test":
        test_log_filename = osp.join(args.experiment_dir, "test_log.txt")
        helper.setup_logging(test_log_filename, args.log_level, "w")
        args.query_generate_dir = (
            osp.join(args.experiment_dir, "query_generate_dir") + "/"
        )
        helper.create_dir(args.query_generate_dir)
        args.vis_gen_dir = osp.join(args.experiment_dir, "vis_gen_dir") + "/"
        helper.create_dir(args.vis_gen_dir)
    else:
        helper.setup_logging(log_filename, args.log_level, "w")

    logging.info("Experiment name: %s and Experiment name 2 %s", exp_name, exp_name_2)
    logging.info("%s", args)

    device, gpu_array = helper.get_device(args)
    args.device = device

    writer = SummaryWriter(log_dir=osp.join(args.experiment_dir, "tb_logs"))

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
    net = Autoencoder(args).to(args.device)
    checkpoint = torch.load(
        args.checkpoint_dir_base + "/" + args.checkpoint + ".pt",
        map_location=args.device,
    )
    net.load_state_dict(checkpoint["model"])
    net.eval()

    # Load latent flow network
    latent_flow_network = latent_flows.get_generator(
        args.emb_dims,
        args.cond_emb_dim,
        device,
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

    if args.train_mode == "test":
        pass
    else:
        optimizer = torch.optim.Adam(latent_flow_network.parameters(), lr=0.00003)
        start_epoch = 0

        # If a checkpoint is provided, intitialize network, optimizer, and
        # schedule with checkpoint configuration.
        if args.latent_load_checkpoint is not None:
            checkpoint_dir = (
                f"{args.new_checkpoint_dir}/{args.latent_load_checkpoint}.pt"
            )
            checkpoint = torch.load(checkpoint_dir, map_location=args.device)
            latent_flow_network.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint["current_epoch"]

        best_loss = np.inf
        for epoch in range(start_epoch, args.epochs):
            loss_meter = helper.AverageMeter()

            if (epoch + 1) % 5 == 0:
                if args.text_query is not None:
                    generate_on_query_text(args, clip_model, net, latent_flow_network)

            train_one_epoch(
                args, latent_flow_network, train_dataloader_new, optimizer, epoch, loss_meter, writer
            )
            val_loss = val_one_epoch(
                args, latent_flow_network, val_dataloader_new, epoch, writer
            )

            filename = f"{args.checkpoint_dir}/last.pt"
            logging.info("Saving Model... %s", filename)
            torch.save(
                {
                    "model": latent_flow_network.state_dict(),
                    "args": args,
                    "current_epoch": epoch,
                },
                filename,
            )

            if best_loss > val_loss:
                best_loss = val_loss
                filename = f"{args.checkpoint_dir}/best.pt"
                logging.info("Saving Model... %s", filename)
                torch.save(
                    {
                        "model": latent_flow_network.state_dict(),
                        "args": args,
                        "current_epoch": epoch,
                    },
                    filename,
                )


if __name__ == "__main__":
    main()
