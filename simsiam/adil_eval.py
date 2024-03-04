# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import random
import shutil
from functools import partial

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torchvision import datasets, models, transforms
from tqdm import tqdm

from SimCLR import distributed as dist_utils
from torch import distributed as dist


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
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=4096,
    type=int,
    metavar="N",
    help="mini-batch size (default: 4096), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial (base) learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
    dest="weight_decay",
)
parser.add_argument(
    "--distributed_mode",
    action="store_true",
    help="Enable distributed training",
)
parser.add_argument("--distributed_launcher", default="slurm")
parser.add_argument("--distributed_backend", default="nccl")
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--pretrained_checkpoint",
    default="",
    type=str,
    help="Path to simsiam pretrained checkpoint.",
)
parser.add_argument("--lars", action="store_true", help="Use LARS")
parser.add_argument(
    "--checkpoint_dir",
    default="",
    help="Checkpoint directory to save eval model checkpoints.",
)


best_acc1 = 0


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

def setup() -> None:
    """Initialize the process group."""
    dist.init_process_group("nccl")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()


def main():
    args = parser.parse_args()
    global best_acc1

    # torch.multiprocessing.set_start_method("spawn")
    if args.distributed_mode:
        # dist_utils.init_distributed_mode(
        #     launcher=args.distributed_launcher,
        #     backend=args.distributed_backend,
        # )
        setup()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.cuda.empty_cache()
        device_id = torch.cuda.current_device()
    else:
        device_id = None

    # create model
    print(f"Creating model {args.arch}")
    model = models.__dict__[args.arch]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained_checkpoint:
        if os.path.isfile(args.pretrained_checkpoint):
            print(f"Loading checkpoint {args.pretrained_checkpoint}")
            checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith("module.encoder") and not k.startswith(
                    "module.encoder.fc"
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        else:
            raise ValueError(f"No checkpoint found at: {args.pretrained_checkpoint}")

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size * 8 / 256

    if args.distributed_mode and dist_utils.is_dist_avail_and_initialized():
        # torch.cuda.set_device(device_id)
        model = model.cuda(device_id)
        model = DDP(model, device_ids=[device_id])
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device_id)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(
        parameters, init_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    if args.lars:
        print("Use LARS optimizer.")
        # from apex.parallel.LARC import LARC
        from LARC import LARC

        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ],
        ),
    )

    if dist_utils.is_dist_avail_and_initialized() and args.distributed_mode:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            seed=args.seed,
        )
    else:
        train_sampler = None

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
        pin_memory=True,  # TODO(arashaf): this was set to false in training script.
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            val_dir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ],
            ),
        ),
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    for epoch in tqdm(range(args.epochs)):
        print(f"Starting training epoch: {epoch}")
        if dist_utils.is_dist_avail_and_initialized():
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device_id, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device_id, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.checkpoint_dir and dist_utils.get_rank() == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_name = "eval_checkpoint_{:04d}.pth.tar".format(epoch)
            checkpoint_file = os.path.join(args.checkpoint_dir, checkpoint_name)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                checkpoint_file,
            )
            if epoch == 0:
                sanity_check(model.state_dict(), args.pretrained_checkpoint)


def train(train_loader, model, criterion, optimizer, epoch, device_id, args):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    for images, target in tqdm(train_loader):
        images = images.cuda(device_id, non_blocking=True)
        target = target.cuda(device_id, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion, device_id, args):
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for images, target in tqdm(val_loader):
            images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        print(
            "Validation Accuracy@1 {top1.avg:.3f}, Accuracy@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    print(f"Saving checkpoint at: {filename}")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print(f"Loading {pretrained_weights} for sanity check")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if "fc.weight" in k or "fc.bias" in k:
            continue

        # name in pretrained model
        k_pre = (
            "module.encoder." + k[len("module.") :]
            if k.startswith("module.")
            else "module.encoder." + k
        )

        assert (
            state_dict[k].cpu() == state_dict_pre[k_pre]
        ).all(), "{} is changed in linear classifier training.".format(k)

    print("Sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr


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


if __name__ == "__main__":
    main()
