"""Utilities for distributed training."""
import os
import subprocess

import torch
import torch.distributed as dist


def init_distributed_mode(
    launcher,
    backend,
) -> None:
    """Launch distributed training based on given launcher and backend.

    Parameters
    ----------
    launcher : {'pytorch', 'slurm'}
        Specifies if pytorch launch utitlity (`torchrun`) is being
        used or if running on a SLURM cluster.
    backend : {'nccl', 'gloo', 'mpi'}
        Specifies which backend to use when initializing a process group.
    """
    if launcher == "pytorch":
        launch_pytorch_dist(backend)
    elif launcher == "slurm":
        launch_slurm_dist(backend)
    else:
        raise RuntimeError(
            f"Invalid launcher type: {launcher}. Use 'pytorch' or 'slurm'.",
        )


def launch_pytorch_dist(backend) -> None:
    """Initialize a distributed process group with PyTorch.

    NOTE: This method relies on `torchrun` to set 'MASTER_ADDR',
    'MASTER_PORT', 'RANK', 'WORLD_SIZE' and 'LOCAL_RANK' as environment variables

    Parameters
    ----------
    backend : {'nccl', 'gloo', 'mpi'}
        Specifies which backend to use when initializing a process group. Can be
        one of ``"nccl"``, ``"gloo"``, or ``"mpi"``.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    disable_non_master_print()  # only print in master process
    dist.barrier()


def launch_slurm_dist(backend) -> None:
    """Initialize a distributed process group when using SLURM.

    Parameters
    ----------
    backend : {'nccl', 'gloo', 'mpi'}
        Specifies which backend to use when initializing a process group. Can be
        one of ``"nccl"``, ``"gloo"``, or ``"mpi"``.
    """
    # set the MASTER_ADDR, MASTER_PORT, RANK and WORLD_SIZE
    # as environment variables before initializing the process group
    if "MASTER_ADDR" not in os.environ:
        node_list = os.environ["SLURM_NODELIST"]
        os.environ["MASTER_ADDR"] = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1",
        )
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29400"
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    local_rank = int(os.environ["SLURM_LOCALID"])
    print(f"Initializing distributed training in process {local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    disable_non_master_print()  # only print on master process
    dist.barrier()


# the following functions were adapted from:
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py
def disable_non_master_print():
    """Disable printing if not master process.

    Notes
    -----
    Printing can be forced by adding a boolean flag, 'force', to the keyword arguments
    to the print function call.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):  # noqa: A001
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """Check if the distributed package is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the total number of processes a distributed process group.

    It returns 1 if the PyTorch distributed package is unavailable or the
    default process group has not been initialized.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the global rank of the current process.

    Returns 0 if the PyTorch distribued package is unavailable or the
    default process group has not been initialized.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if the current process is the Master proces.

    The master process typically has a rank of 0.
    """
    return not is_dist_avail_and_initialized() or get_rank() == 0
