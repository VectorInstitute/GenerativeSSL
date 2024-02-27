# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import random
from datetime import datetime
from functools import partial

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models
from tqdm import tqdm

from SimCLR import distributed as dist_utils
from simsiam import builder, loader


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--data_dir",
    metavar="DIR",
    default="/scratch/ssd004/datasets/imagenet256",
    help="path to dataset.",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--num_workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 512), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.05,
    type=float,
    metavar="LR",
    help="initial (base) learning rate",
    dest="lr",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--resume_from_checkpoint",
    default="",
    type=str,
    help="Path to latest checkpoint.",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)

# simsiam specific configs:
parser.add_argument(
    "--dim", default=2048, type=int, help="feature dimension (default: 2048)"
)
parser.add_argument(
    "--pred-dim",
    default=512,
    type=int,
    help="hidden dimension of the predictor (default: 512)",
)
parser.add_argument(
    "--fix-pred-lr", action="store_true", help="Fix learning rate for the predictor"
)

parser.add_argument(
    "--distributed_mode",
    action="store_true",
    help="Enable distributed training",
)
parser.add_argument("--distributed_launcher", default="slurm")
parser.add_argument("--distributed_backend", default="nccl")
parser.add_argument(
    "--checkpoint_dir",
    default="/projects/imagenet_synthetic/model_checkpoints",
    help="Checkpoint root directory.",
)
parser.add_argument(
    "--experiment",
    default="",
    help="Experiment name.",
)
parser.add_argument(
    "--use_synthetic_data",
    action=argparse.BooleanOptionalAction,
    help="Whether to use real data or synthetic data for training.",
)
parser.add_argument(
    "--synthetic_data_dir",
    default="/projects/imagenet_synthetic/synthetic_icgan",
    help="Path to the root of synthetic data.",
)
parser.add_argument(
    "--synthetic_index_min",
    default=0,
    type=int,
    help="Synthetic data files are named filename_i.JPEG. This index determines the lower bound for i.",
)
parser.add_argument(
    "--synthetic_index_max",
    default=9,
    type=int,
    help="Synthetic data files are named filename_i.JPEG. This index determines the upper bound for i.",
)
parser.add_argument(
    "--generative_augmentation_prob",
    default=None,
    type=float,
    help="The probability of applying a generative model augmentation to a view. Applies to the views separately.",
)


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int) -> None:
    """Initialize worker processes with a random seed.

    Parameters
    ----------
    worker_id : int
        ID of the worker process.
    num_workers : int
        Total number of workers that will be initialized.
    rank : int
        The rank of the current process.
    seed : int
        A random seed used determine the worker seed.
    """
    worker_seed = num_workers * rank + worker_id + seed
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)


def main():
    args = parser.parse_args()
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    checkpoint_subdir = (
        f"{args.experiment}_{current_time}" if args.experiment else f"{current_time}"
    )
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, checkpoint_subdir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(args)

    torch.multiprocessing.set_start_method("spawn")
    if args.distributed_mode:
        dist_utils.init_distributed_mode(
            launcher=args.distributed_launcher,
            backend=args.distributed_backend,
        )
        device_id = torch.cuda.current_device()
    else:
        device_id = None

    # Data loading.
    if args.use_synthetic_data:
        print(
            f"Using synthetic data for training at {args.synthetic_data_dir} between indices {args.synthetic_index_min} and {args.synthetic_index_max}."
        )
        train_dataset = loader.ImageNetSynthetic(
            args.data_dir,
            args.synthetic_data_dir,
            index_min=args.synthetic_index_min,
            index_max=args.synthetic_index_max,
            generative_augmentation_prob=args.generative_augmentation_prob,
        )
    else:
        print(f"Using real data for training at {args.data_dir}.")
        train_data_dir = os.path.join(args.data_dir, "train")
        train_dataset = datasets.ImageFolder(train_data_dir, loader.TwoCropsTransform())

    train_sampler = None
    if dist_utils.is_dist_avail_and_initialized() and args.distributed_mode:
        train_sampler = DistributedSampler(
            train_dataset,
            seed=args.seed,
            drop_last=True,
        )
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_workers,
        rank=dist_utils.get_rank(),
        seed=args.seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        worker_init_fn=init_fn,
        pin_memory=False,
        drop_last=True,
    )

    print(f"Creating model {args.arch}")
    model = builder.SimSiam(models.__dict__[args.arch], args.dim, args.pred_dim)

    if args.distributed_mode and dist_utils.is_dist_avail_and_initialized():
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # set the single device scope, otherwise DistributedDataParallel will
        # use all available devices
        torch.cuda.set_device(device_id)
        model = model.cuda(device_id)
        model = DDP(model, device_ids=[device_id])
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model)  # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(device_id)

    if args.fix_pred_lr:
        optim_params = [
            {"params": model.module.encoder.parameters(), "fix_lr": False},
            {"params": model.module.predictor.parameters(), "fix_lr": True},
        ]
    else:
        optim_params = model.parameters()

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256.0
    optimizer = torch.optim.SGD(
        optim_params,
        init_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    # Optionally resume from a checkpoint
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            print(f"Loading checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Loaded checkpoint {args.resume_from_checkpoint} successfully.")
        else:
            raise ValueError(f"No checkpoint found at: {args.resume_from_checkpoint}")

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting training epoch: {epoch}")
        if dist_utils.is_dist_avail_and_initialized():
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device_id, args)

        # Checkpointing.
        if dist_utils.get_rank() == 0:
            checkpoint_name = f"checkpoint_{epoch}.pth.tar"
            checkpoint_file = os.path.join(args.checkpoint_dir, checkpoint_name)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename=checkpoint_file,
            )


def train(train_loader, model, criterion, optimizer, epoch, device_id, args):
    """Single epoch training code."""
    # switch to train mode
    model.train()

    # for i, (images, _) in enumerate(train_loader):
    for images, _ in tqdm(train_loader):
        images[0] = images[0].cuda(device_id, non_blocking=True)
        images[1] = images[1].cuda(device_id, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save state dictionary into a model checkpoint."""
    print(f"Saving checkpoint at: {filename}")
    torch.save(state, filename)


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule."""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if "fix_lr" in param_group and param_group["fix_lr"]:
            param_group["lr"] = init_lr
        else:
            param_group["lr"] = cur_lr


if __name__ == "__main__":
    main()
