from torchvision import datasets, transforms
from torchvision.transforms import transforms

from SimCLR.exceptions.exceptions import InvalidDatasetSelection
from SimCLR.datasets.data_aug.center_crop import CostumeCenterCrop

class SupervisedDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform(size):
        """Return a set of simple transformations for supervised learning.

        Args:
            size (int): Image size.
        """
        transform_list = [
            CostumeCenterCrop(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transform_list)

    
    def get_dataset(self, name, train = True):
        if name == "imagenet":
            if train:
                split = "train"
            else:
                split = "val"
            return datasets.ImageNet(
                self.root_folder,
                split=split,
                transform=self.get_transform(224),
            )
        elif name == "cifar10":
            return datasets.CIFAR10(
                self.root_folder,
                train=train,
                transform= self.get_transform(32),
                download=True,
            )
        elif name == "stl10":
            if train:
                split = "train"
            else:
                split = "test"
            return datasets.STL10(
                self.root_folder,
                split=split,
                transform=self.get_transform(96),
                download=True,
            )
        else:
            raise InvalidDatasetSelection()