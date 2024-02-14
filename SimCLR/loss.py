"""SimCLR two-view loss functions."""
import torch
import torch.distributed as dist
from torch import nn


LARGE_NUM = 1e9


# Implementation of GatherLayer from https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py.
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class SimCLRContrastiveLoss(nn.Module):
    """SimCLR contrastive loss implementation for DDP training.

    This is the pytorch version of the original SimCLR loss implementation
    at https://github.com/google-research/simclr/blob/master/tf2/objective.py#L35.

    Parameters
    ----------
    temperature : float
        Temperature parameter for the contrastive loss.

    """

    def __init__(self, temperature: float = 1.0) -> None:
        super(SimCLRContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        hidden1: torch.Tensor,
        hidden2: torch.Tensor,
        device_id,
        l2_normalize=True,
    ) -> torch.Tensor:
        world_size = dist.get_world_size()
        batch_size = hidden1.shape[0]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if world_size > 1:
            hidden1_large = torch.cat(GatherLayer.apply(hidden1), dim=0)
            hidden2_large = torch.cat(GatherLayer.apply(hidden2), dim=0)

            enlarged_batch_size = batch_size * world_size

            global_rank = dist.get_rank()

            labels_idx = torch.arange(batch_size) + global_rank * batch_size
            labels = nn.functional.one_hot(labels_idx, enlarged_batch_size * 2)
            mask = nn.functional.one_hot(labels_idx, enlarged_batch_size)
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = nn.functional.one_hot(torch.arange(batch_size), batch_size * 2)
            mask = nn.functional.one_hot(torch.arange(batch_size), batch_size)

        labels = labels.float().cuda(device_id)
        mask = mask.cuda(device_id)

        if l2_normalize:
            hidden1 = nn.functional.normalize(hidden1, dim=-1)
            hidden2 = nn.functional.normalize(hidden2, dim=-1)
            hidden1_large = nn.functional.normalize(hidden1_large, dim=-1)
            hidden2_large = nn.functional.normalize(hidden2_large, dim=-1)

        logits_aa = torch.matmul(hidden1, hidden1_large.T) / self.temperature
        logits_aa = logits_aa - mask * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T) / self.temperature
        logits_bb = logits_bb - mask * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T) / self.temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.T) / self.temperature

        criterion = nn.CrossEntropyLoss(reduction="sum")
        loss_a = criterion(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = criterion(torch.cat([logits_ba, logits_bb], 1), labels)
        return loss_a + loss_b
