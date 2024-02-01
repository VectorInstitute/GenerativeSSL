from torchvision import datasets, transforms
from torchvision.transforms import transforms

from SimCLR.data_aug.gaussian_blur import GaussianBlur
from SimCLR.data_aug.rcdm_aug import RCDMInference
from SimCLR.data_aug.icgan_aug import ICGANInference
from SimCLR.data_aug.rcdm_config import get_config
from SimCLR.data_aug.icgan_config import get_icgan_config
from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator
from SimCLR.exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1, rcdm_agumentation=False, icgan_agumentation=False, device_id=None):
        """Return a set of data augmentation transformations as described in the SimCLR paper.

        Args:
            size (int): Image size.
            s (float, optional): Magnitude of the color distortion. Defaults to 1.
            rcdm_agumentation (bool, optional): Whether to use RCDM augmentation. Defaults to True.
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transform_list = [
            transforms.Resize(size=(size,size)),
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(size=(size, size)),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
        ]
        if rcdm_agumentation:
            rcdm_config = get_config()
            transform_list.insert(2, RCDMInference(rcdm_config, device_id))

        elif icgan_agumentation:
            icgan_config = get_icgan_config()
            transform_list.insert(2, ICGANInference(icgan_config, device_id))

        return transforms.Compose(transform_list)

    def get_dataset(self, name, n_views, rcdm_agumentation=False, icgan_agumentation=False, device_id=None):
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        32, rcdm_agumentation=rcdm_agumentation,
                        icgan_agumentation=icgan_agumentation,
                        device_id=device_id
                    ),
                    n_views,
                ),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        96, rcdm_agumentation=rcdm_agumentation,
                        icgan_agumentation=icgan_agumentation,
                        device_id=device_id
                    ),
                    n_views,
                ),
                download=True,
            ),
            "imagenet": lambda: datasets.ImageNet(
                self.root_folder,
                split="train",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        224, rcdm_agumentation=rcdm_agumentation,
                        icgan_agumentation=icgan_agumentation,
                        device_id=device_id
                    ),
                    n_views,
                ),
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
