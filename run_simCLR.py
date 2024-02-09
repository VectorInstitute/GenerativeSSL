import argparse
import random
from functools import partial

import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data.distributed import DistributedSampler
from torchvision import models

from SimCLR import distributed as dist_utils
from SimCLR.data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from SimCLR.models.resnet_simclr import ResNetSimCLR
from SimCLR.simclr import SimCLR


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch SimCLR")
parser.add_argument(
    "-data",
    metavar="DIR",
    default="/scratch/ssd004/datasets/imagenet256",
    help="path to dataset, for imagenet: /scratch/ssd004/datasets/imagenet256 ",
)
parser.add_argument(
    "-dataset-name",
    default="imagenet",
    help="dataset-name",
    choices=["stl10", "cifar10", "imagenet"],
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--num_workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0003,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
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
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--fp16-precision",
    action="store_true",
    help="Whether or not to use 16-bit precision GPU training.",
)

parser.add_argument(
    "--out_dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--log-every-n-steps", default=100, type=int, help="Log every n steps"
)
parser.add_argument(
    "--temperature",
    default=0.07,
    type=float,
    help="softmax temperature (default: 0.07)",
)
parser.add_argument(
    "--n-views",
    default=2,
    type=int,
    metavar="N",
    help="Number of views for contrastive learning training.",
)
parser.add_argument(
    "--rcdm_augmentation", action="store_true", help="Use RCDM augmentation or not."
)
parser.add_argument(
    "--icgan_augmentation", action="store_true", help="Use ICGAN augmentation or not."
)
parser.add_argument(
    "--distributed_mode", action="store_true", help="Enable distributed training"
)
parser.add_argument("--distributed_launcher", default="slurm")
parser.add_argument("--distributed_backend", default="nccl")


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
    print(args)

    torch.multiprocessing.set_start_method("spawn")

    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."

    if args.distributed_mode:
        dist_utils.init_distributed_mode(
            launcher=args.distributed_launcher,
            backend=args.distributed_backend,
        )
        device_id = torch.cuda.current_device()
    else:
        device_id = None

    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(
        args.dataset_name,
        args.n_views,
        args.rcdm_augmentation,
        args.icgan_augmentation,
        device_id,
    )
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

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    if args.distributed_mode and dist_utils.is_dist_avail_and_initialized():
        # set the single device scope, otherwise DistributedDataParallel will
        # use all available devices
        torch.cuda.set_device(device_id)
        model = model.cuda(device_id)
        model = DDP(model, device_ids=[device_id])
    else:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    simclr = SimCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device_id=device_id,
        args=args,
    )
    simclr.train(train_loader)


if __name__ == "__main__":
    main()
