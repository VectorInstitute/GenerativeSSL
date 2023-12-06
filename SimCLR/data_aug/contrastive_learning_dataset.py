from torchvision import datasets, transforms
from torchvision.transforms import transforms

from SimCLR.data_aug.gaussian_blur import GaussianBlur
from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator
from SimCLR.exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper.
        
        Args:
            size (int): Image size.
            s (float, optional): Magnitude of the color distortion. Defaults to 1.
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transform_list = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
        ]

        data_transforms = transforms.Compose(transform_list)
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32), n_views
                ),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                download=True,
            ),
            "imagenet": lambda: datasets.ImageNet(
                self.root_folder,
                split="train",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(224), n_views
                ),
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
