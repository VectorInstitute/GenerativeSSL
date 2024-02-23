import argparse
import os
import time

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverSinglestepScheduler,
    StableUnCLIPImg2ImgPipeline,
)
from torchvision import datasets, transforms

from icgan.data_utils import utils as data_utils


class StableGenerator(object):
    def __init__(self, opt):
        self.opt = opt
        # model
        self.model = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip-small",
            torch_dtype=torch.float16,
            variation="fp16",
        )

        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
        self.model.to(device)
        print(f"Using device: {device}")

        if opt.dpm:
            self.model.scheduler = DPMSolverSinglestepScheduler.from_config(
                self.model.scheduler.config, rescale_betas_zero_snr=True
            )
        else:
            self.model.scheduler = DDIMScheduler.from_config(
                self.model.scheduler.config,
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
            )

        print("Scheduler:", self.model.scheduler)

        # image size
        self.height = self.opt.img_save_size
        self.width = self.opt.img_save_size

        # inference steps
        self.num_inference_steps = self.opt.steps

        # eta (0, 1)
        self.eta = self.opt.ddim_eta

        self.generator = torch.Generator()
        self.generator.manual_seed(self.opt.image_version)

    def generate(self, input_image, n_sample_per_image=10):
        trans = transforms.Resize(size=(self.height, self.width))
        synth_images = self.model(
            input_image,
            eta=self.eta,
            num_images_per_prompt=n_sample_per_image,
            num_inference_steps=self.num_inference_steps,
            generator=self.generator,
        ).images
        return [trans(img) for img in synth_images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--img_save_size", type=int, default=224, help="image saving size"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--dpm",
        action="store_true",
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="use ddim sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for inference",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="Number of shards to split the dataset.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        help="Index of the shard",
    )
    parser.add_argument(
        "--image_version",
        type=int,
        help="Version and seed for generated images.",
    )
    parser.add_argument(
        "--counter",
        type=int,
        help="Counter.",
    )
    opt = parser.parse_args()

    if opt.outdir is not None:
        os.makedirs(opt.outdir, exist_ok=True)

    transform_list = [
        data_utils.CenterCropLongEdge(),
        transforms.Resize(size=(opt.img_save_size, opt.img_save_size)),
    ]
    transform = transforms.Compose(transform_list)

    imagenet_dataset = datasets.ImageNet(
        "/scratch/ssd004/datasets/imagenet256", split="train", transform=transform
    )

    Stable_generator = StableGenerator(opt)
    n = len(imagenet_dataset)

    counter = 0
    for i in range(n):
        counter += 1
        if counter > opt.counter:
            break
        if i % opt.num_shards == opt.shard_index:
            batch = imagenet_dataset[i]
            image = batch[0]
            start = time.time()
            generated_images = Stable_generator.generate(
                image,
                n_sample_per_image=1,
            )
            end = time.time()
            # print(f"Time taken to generate images: {end-start} seconds")

            # Save images.
            path = imagenet_dataset.samples[i][0]
            start = time.time()
            _save_images(path, generated_images, opt.outdir, opt.image_version)
            end = time.time()
            print(f"Time taken to save images: {end-start} seconds")


def _save_images(path, images, out_dir, image_version):
    # get the class name
    out_folder = path.split("/")[-1].split(".")[0].split("_")[0]
    # get the (class name_image number)
    file_name = path.split("/")[-1].split(".")[0]
    # create a folder for each class
    save_folder = os.path.join(out_dir, out_folder)
    print(f"Path: {path}")
    print(f"save_folder: {save_folder}")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    for _, img in enumerate(images):
        save_file = os.path.join(save_folder, f"{file_name}_{image_version}.JPEG")
        img.save(save_file)


if __name__ == "__main__":
    main()
