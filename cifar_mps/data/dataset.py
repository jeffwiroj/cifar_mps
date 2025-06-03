import torch
import torchvision as tv
from dataclasses import dataclass
from cifar_mps.data.transformations import get_transformation

dataset_mean = (0.4914, 0.4822, 0.4465)
dataset_std = (0.2023, 0.1994, 0.2010)


def download_dataset(data_dir="data/"):
    """
    Downloads the CIFAR-10 dataset and saves it to the directory data_dir

    Args:
        The directory where the CIFAR-10 dataset should be saved.
    """
    tv.datasets.CIFAR10(root=data_dir, train=True, download=True)
    tv.datasets.CIFAR10(root=data_dir, train=False, download=True)


def get_datasets(config: dataclass, data_dir="data/"):
    train_transformations = get_transformation(config.augmentations)
    test_transformations = get_transformation("test")
    train_set = tv.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transformations
    )
    test_set = tv.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transformations
    )

    return train_set, test_set


def get_dataloaders(config, data_dir="data/"):

    train_set, test_set = get_datasets(config, data_dir)

    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, drop_last=False, pin_memory=pin_memory
    )

    return train_loader, test_loader
