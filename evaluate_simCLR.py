import argparse
import random
from functools import partial

import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data.distributed import DistributedSampler
from torchvision import models

from SimCLR import distributed as dist_utils
from SimCLR.data_aug.supervised_dataset import SupervisedDataset
from SimCLR.models.resnet_pretrained import PretrainedResNet
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
    "--seed", default=42, type=int, help="seed for initializing training. "
)

parser.add_argument(
    "--distributed_mode", action="store_true", help="Enable distributed training"
)
parser.add_argument("--distributed_launcher", default="slurm")
parser.add_argument("--distributed_backend", default="nccl")
parser.add_argument("--pretrained_model_file", default=None, help="Path to the pretrained model file.")
parser.add_argument("--linear_evaluation", 
                    action="store_true",
                    help="Whether or not to evaluate the linear evaluation of the model.")
parser.add_argument(
    "--log-every-n-steps", default=100, type=int, help="Log every n steps"
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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    args = parser.parse_args()
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

    dataset = SupervisedDataset(args.data)
    train_dataset = dataset.get_dataset(
        name = args.dataset_name,
        train=True,
    )
    test_dataset = dataset.get_dataset(
        name = args.dataset_name,
        train=False,
    )
    train_sampler = None
    test_sampler = None

    if dist_utils.is_dist_avail_and_initialized() and args.distributed_mode:
            train_sampler = DistributedSampler(
                train_dataset,
                seed=args.seed,
                drop_last=True,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                seed=args.seed,
                drop_last=False,
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        num_workers=args.num_workers,
        worker_init_fn=init_fn,
        pin_memory=False,
        drop_last=False,
    )
    if args.dataset_name == "cifar10":
        num_classes = 10
    elif args.dataset_name == "stl10":
        num_classes = 10
    elif args.dataset_name == "imagenet":
        num_classes = 1000

    model = PretrainedResNet(base_model=args.arch, pretrained_dir = args.pretrained_model_file, linear_eval=args.linear_evaluation, num_classes=num_classes)

    if args.distributed_mode and dist_utils.is_dist_avail_and_initialized():
        # set the single device scope, otherwise DistributedDataParallel will
        # use all available devices
        torch.cuda.set_device(device_id)
        model = model.cuda(device_id)
        model = DDP(model, device_ids=[device_id])
    else:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().cuda(device_id)

    n_iter = 0

    for epoch in range(args.epochs):
        if dist_utils.is_dist_avail_and_initialized():
            train_loader.sampler.set_epoch(epoch)
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.cuda(device_id)
            y_batch = y_batch.cuda(device_id)

            logits = model(x_batch)
            print(y_batch)
            print(logits.shape)
            print(logits)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if n_iter % args.log_every_n_steps == 0:
                top1_train_accuracy /= counter + 1
                top1_accuracy = 0
                top5_accuracy = 0
                for counter, (x_batch, y_batch) in enumerate(test_loader):
                    x_batch = x_batch.cuda(device_id)
                    y_batch = y_batch.cuda(device_id)

                    logits = model(x_batch)

                    top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
                    top1_accuracy += top1[0]
                    top5_accuracy += top5[0]

                top1_accuracy /= counter + 1
                top5_accuracy /= counter + 1
                print(
                    f"Epoch {epoch}\t Iter {n_iter}\t Top1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}",
                )
            n_iter += 1

   


if __name__ == "__main__":
    main()