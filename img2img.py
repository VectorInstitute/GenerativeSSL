import argparse
import torch
from diffusers import StableUnCLIPImg2ImgPipeline, DDIMScheduler, DPMSolverSinglestepScheduler
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import json



class StableGenerator(object):
    def __init__(self, opt):
        self.opt = opt
        # model
        self.model = StableUnCLIPImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16, variation="fp16"
                )

        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
        self.model.to(device)
        print(f"Using device: {device}")

        if opt.dpm:
            self.model.scheduler = DPMSolverSinglestepScheduler.from_config(self.model.scheduler.config, rescale_betas_zero_snr=True)
        else:
            self.model.scheduler = DDIMScheduler.from_config(self.model.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")

        print("Scheduler:", self.model.scheduler)

        # self.model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # self.model.vae.enable_xformers_memory_efficient_attention(attention_op=None)

        # image size
        self.height = self.opt.img_save_size
        self.width = self.opt.img_save_size

        # inference steps
        self.num_inference_steps = self.opt.steps

        # eta (0, 1)
        self.eta = self.opt.ddim_eta
    
    def generate(self, input_image, n_sample_per_image=10):
        transfoem_2 = transforms.Resize(size=(self.height, self.width))
        synth_images = self.model(input_image, eta=self.eta, num_images_per_prompt=n_sample_per_image, num_inference_steps=self.num_inference_steps).images
        synth_images = [transfoem_2(img) for img in synth_images]
        return synth_images



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--img_save_size",
        type=int,
        default=224,
        help="image saving size"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim",
        action='store_true',
        help="use ddim sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for inference",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="start index",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="end index",
    )
    opt = parser.parse_args()

    # s = 1
    # size = 128
    # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    if opt.outdir is not None:
        os.makedirs(opt.outdir, exist_ok=True)

    transform_list = [
        transforms.Resize(size=(opt.img_save_size,opt.img_save_size)),
        # transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform_list)

    imagenet_dataset = datasets.ImageNet("/local/ssd/m2kowal/imagenet", split="train", transform = transform)
    # root_folder = "/local/ssd/m2kowal/imagenet/"

    Stable_generator = StableGenerator(opt)
    n = len(imagenet_dataset) 
    assert opt.start < n
    assert opt.end <= n   

    for i in range(opt.start, opt.end):
        print(f"Batch {i}")
        batch = imagenet_dataset[i]
        images = batch[0]
        start = time.time()
        generated_images = Stable_generator.generate(images, n_sample_per_image=opt.n_samples)
        end = time.time()
        print(f"Time taken to generate images: {end-start} seconds")

        ## save images
        path = imagenet_dataset.samples[i][0]
        start = time.time()
        save_images(path, generated_images, opt.outdir) 
        end = time.time()
        print(f"Time taken to save images: {end-start} seconds")

def save_images(path, images, out_dir):
    for j, img in enumerate(images):
        out_folder = path.split("/")[-1].split(".")[0].split("_")[0] # get the class name
        file_name = path.split("/")[-1].split(".")[0] # get the (class name_image number)
        save_folder = os.path.join(out_dir, out_folder) # create a folder for each class

        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        
        for j, img in enumerate(images):
            save_file = os.path.join(save_folder, f"{file_name}_{j+1}.png")
            img.save(save_file)

if __name__ == "__main__":
    main()