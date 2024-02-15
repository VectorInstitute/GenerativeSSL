import os

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SimCLR import distributed as dist_utils
from SimCLR import loss

from .utils import save_checkpoint, save_config_file
from torch.profiler import profile, record_function, ProfilerActivity


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"]
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.device_id = kwargs["device_id"]
        self.writer = SummaryWriter()
        self.criterion = loss.SimCLRContrastiveLoss(self.args.temperature).cuda(
            self.device_id
        )

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        print(f"Start SimCLR training for {self.args.epochs} epochs.")

        for epoch_counter in tqdm(range(self.args.epochs), desc="Training Progress"):
            if dist_utils.is_dist_avail_and_initialized():
                train_loader.sampler.set_epoch(epoch_counter)
            for images, _ in tqdm(train_loader):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                         record_shapes=True,
                         with_stack=True,
                         profile_memory=True) as prof:

                    with record_function("forward_pass"):
                        view1_images = images["view1"].cuda(self.device_id)  # noqa: PLW2901
                        view2_images = images["view2"].cuda(self.device_id)  # noqa: PLW2901
                        # Concatenate the two views so we run inference once.
                        images = torch.cat([view1_images, view2_images], dim=0)  # noqa: PLW2901
                        images = images.cuda(self.device_id)  # noqa: PLW2901
                    with record_function("compute_loss"), autocast(enabled=self.args.fp16_precision):
                        features = self.model(images)
                        hidden1, hidden2 = torch.split(features, features.shape[0] // 2)
                        loss = self.criterion(hidden1, hidden2, self.device_id)
                    with record_function("backward_pass"):
                        self.optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    print(
                        f"Calculating loss at iteration: {n_iter}, loss: {loss}",
                    )
                    self.writer.add_scalar("loss", loss, global_step=n_iter)
                    self.writer.add_scalar(
                        "learning_rate",
                        self.scheduler.get_last_lr()[0],
                        global_step=n_iter,
                    )
                    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            print(f"Epoch: {epoch_counter}\tLoss: {loss}")
            # save model checkpoints
            checkpoint_name = "checkpoint_{:04d}.pth.tar".format(self.args.epochs)
            save_checkpoint(
                {
                    "epoch": self.args.epochs,
                    "arch": self.args.arch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best=False,
                filename=os.path.join(self.writer.log_dir, checkpoint_name),
            )
            print(
                f"Model checkpoint and metadata has been saved at {self.writer.log_dir}."
            )

        print("Training has finished.")