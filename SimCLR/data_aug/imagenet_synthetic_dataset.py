"""ImageNet synthetic dataset."""

import os
import random

import torch
from PIL import Image
from torchvision import datasets, transforms

from SimCLR.data_aug.gaussian_blur import GaussianBlur


def _get_simclr_transforms(size, random_crop=False, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper.

    Args:
        size (int): Image size.
        s (float, optional): Magnitude of the color distortion. Defaults to 1.
    """
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
    ]
    if random_crop:
        transform_list.insert(0, transforms.RandomResizedCrop(size=size))

    return transforms.Compose(transform_list)


class ImageNetSynthetic(datasets.ImageNet):
    def __init__(
        self,
        imagenet_root,
        imagenet_synthetic_root,
        index_min=0,
        index_max=9,
        generative_augmentation_prob=None,
        load_one_real_image=False,
        split="train",
        synthetic_transforms=None,
        real_transforms=None,
    ):
        super(ImageNetSynthetic, self).__init__(
            root=imagenet_root,
            split=split,
        )
        self.imagenet_root = imagenet_root
        self.imagenet_synthetic_root = imagenet_synthetic_root
        self.index_min = index_min
        self.index_max = index_max
        self.generative_augmentation_prob = generative_augmentation_prob
        self.load_one_real_image = load_one_real_image
        if synthetic_transforms is not None:
            self.synthetic_transforms = synthetic_transforms
        else:
            self.synthetic_transforms = _get_simclr_transforms(size=224)
        
        if real_transforms is not None:
            self.real_transforms = real_transforms
        else:
            self.real_transforms = _get_simclr_transforms(size=224, random_crop=True)
        self.split = split

    def __getitem__(self, index):
        imagenet_filename, label = self.imgs[index]

        def _synthetic_image(filename):
            rand_int = random.randint(self.index_min, self.index_max)
            filename_and_extension = filename.split("/")[-1]
            filename_parent_dir = filename.split("/")[-2]
            image_path = os.path.join(
                self.imagenet_synthetic_root,
                self.split,
                filename_parent_dir,
                filename_and_extension.split(".")[0] + f"_{rand_int}.JPEG",
            )
            return Image.open(image_path).convert("RGB")

        if self.generative_augmentation_prob is not None:
            if torch.rand(1) < self.generative_augmentation_prob:
                # Generate a synthetic image.
                image1 = _synthetic_image(imagenet_filename)
                image1 = self.synthetic_transforms(image1)
            else:
                image1 = self.loader(os.path.join(self.root, imagenet_filename))
                image1 = self.real_transforms(image1)

            if torch.rand(1) < self.generative_augmentation_prob:
                # Generate another synthetic image.
                image2 = _synthetic_image(imagenet_filename)
                image2 = self.synthetic_transforms(image2)
            else:
                image2 = self.loader(os.path.join(self.root, imagenet_filename))
                image2 = self.real_transforms(image2)
        else:
            if self.load_one_real_image:
                image1 = self.loader(os.path.join(self.root, imagenet_filename))
                image1 = self.real_transforms(image1)
            else:
                image1 = _synthetic_image(imagenet_filename)
                image1 = self.synthetic_transforms(image1)
            # image2 is always synthetic.
            image2 = _synthetic_image(imagenet_filename)
            image2 = self.synthetic_transforms(image2)
        
        return index, [image1[0], image2[0]], label