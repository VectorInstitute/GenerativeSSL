# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import torch
from PIL import Image, ImageFilter
from torchvision import datasets, transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
_real_augmentations = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _normalize,
]


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self):
        self.base_transform = transforms.Compose(_real_augmentations)

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


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
        self.real_transforms = transforms.Compose(_real_augmentations)
        # Remove random crop for synthetic image augmentation.
        self.synthetic_transforms = transforms.Compose(_real_augmentations[1:])
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

        return [image1, image2], label
