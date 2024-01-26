import os
from datetime import datetime

import torch
import torch.nn.functional as F  # noqa: N812
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from SimCLR import distributed as dist_utils

from .utils import accuracy, save_checkpoint, save_config_file


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"]
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.device_id = kwargs["device_id"]
        log_dir = os.path.join(self.args.model_dir, self.args.experiment_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.device_id)

    def simclr_logits_and_labels(self, features):
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for i in range(self.args.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda(self.device_id)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).cuda(
            self.device_id
        )
        labels = labels[~mask].view(similarity_matrix.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.device_id)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        print(f"Start SimCLR training for {self.args.epochs} epochs.")

        for epoch_counter in range(self.args.epochs):
            if dist_utils.is_dist_avail_and_initialized():
                train_loader.sampler.set_epoch(epoch_counter)
            # for images, _ in tqdm(train_loader):
            for images, _ in train_loader:
                now_time = datetime.now().strftime("%H:%M:%S")
                print(f"{now_time} - Starting batch iteration: {n_iter}")
                images = torch.cat(images, dim=0)  # noqa: PLW2901
                images = images.cuda(self.device_id)  # noqa: PLW2901

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.simclr_logits_and_labels(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    print(
                        f"Calculating accuracy/loss at iteration: {n_iter}, loss: {loss},acc: top1 - {top1[0]}, top5 - {top5[0]}",
                    )
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar("loss", loss, global_step=n_iter)
                    self.writer.add_scalar("acc/top1", top1[0], global_step=n_iter)
                    self.writer.add_scalar("acc/top5", top5[0], global_step=n_iter)
                    self.writer.add_scalar(
                        "learning_rate",
                        self.scheduler.get_last_lr()[0],
                        global_step=n_iter,
                    )
                    # save model checkpoints
                    checkpoint_name = "checkpoint_{:04d}.pth.tar".format(self.args.epochs)
                    save_checkpoint(
                        {
                            "n_iter": n_iter,
                            "arch": self.args.arch,
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                        },
                        is_best=False,
                        filename=os.path.join(self.writer.log_dir, checkpoint_name),
                    )

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            print(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        print("Training has finished.")

        print(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
