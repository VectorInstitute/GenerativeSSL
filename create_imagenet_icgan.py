import argparse
from torchvision import datasets, transforms
import time
import os
from icgan.config import get_config
from icgan.icgan_inference import ICGANInference
from pytorch_pretrained_biggan import convert_to_images
from icgan.data_utils import utils as data_utils
import torch

import numpy as np

def save(out, torch_format=True):
    if torch_format:
        with torch.no_grad():
            out = out.cpu().numpy()
    img = convert_to_images(out)[0]
    return img

def save_images(path, images, out_dir):
    for j, img in enumerate(images):
        out_folder = path.split("/")[-1].split(".")[0].split("_")[0] # get the class name
        file_name = path.split("/")[-1].split(".")[0] # get the (class name_image number)
        save_folder = os.path.join(out_dir, out_folder) # create a folder for each class

        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        for j, img in enumerate(images):
            save_file = os.path.join(save_folder, f"{file_name}_{j+1}.png")
            img = save(img[np.newaxis, ...], torch_format=False)
            img.save(save_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="/projects/imagenet_synthetic/synthetic_icgan"
    )
    # parser.add_argument(
    #     "--img_save_size",
    #     type=int,
    #     default=224,
    #     help="image saving size"
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=1,
    #     help="batch size",
    # )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="start index",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="end index",
    )
    args = parser.parse_args()

    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)

    # Initialize the ICGANInference class
    config = get_config()
    icgan_inference = ICGANInference(config)

    transform = transforms.Compose([
        data_utils.CenterCropLongEdge(),
        # transforms.Resize((args.img_save_size, args.img_save_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.norm_mean, config.norm_std)])


    imagenet_dataset = datasets.ImageNet("/scratch/ssd004/datasets/imagenet256", split="train", transform=transform)

    n = len(imagenet_dataset)
    if args.end == -1:
        args.end = n
    assert args.start < n
    assert args.end <= n

    for i in range(args.start, args.end):
        print(f"Batch {i}")
        batch = imagenet_dataset[i]
        images = batch[0]
        start = time.time()
        generated_images = icgan_inference.run_inference(input_image_tensor=images.unsqueeze(0))
        end = time.time()
        print(f"Time taken to generate images: {end-start} seconds")

        ## save images
        path = imagenet_dataset.samples[i][0]
        start = time.time()
        save_images(path, generated_images, args.outdir)
        end = time.time()
        print(f"Time taken to save images: {end-start} seconds")

if __name__ == "__main__":
    main()