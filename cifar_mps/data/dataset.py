import torch
import torchvision as tv
from dataclasses import dataclass
from cifar_mps.data.transformations import get_transformation
import matplotlib.pyplot as plt
import numpy as np

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
        test_set,
        batch_size=2 * config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def unnormalize(image, mean, std):
    """
    Undo normalization of an image tensor.
    Args:
        image (torch.Tensor): The image tensor.
        mean (tuple): The mean values used for normalization.
        std (tuple): The std values used for normalization.

    Returns:
        torch.Tensor: The unnormalized image.
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add mean to undo normalization.
    return image


def visualize_images(
    dataloader, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), num_images=8
):
    """
    Visualizes a batch of images from a given DataLoader.
    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the image batches.
        mean (tuple): The mean values used for normalization.
        std (tuple): The std values used for normalization.
        num_images (int): The number of images to display.
    """
    # Get a batch of images
    images, labels = next(iter(dataloader))

    # Unnormalize the images
    images = unnormalize(images, mean, std)

    # Convert the image tensor to a NumPy array (for plotting)
    images = images.numpy().transpose((0, 2, 3, 1))  # Shape (N, C, H, W) -> (N, H, W, C)
    images = np.clip(images, 0, 1)  # Clip values to be between 0 and 1 for displaying

    # Create a grid of images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis("off")  # Hide axes
        ax.set_title(f"Label: {labels[i].item()}")  # Show the label

    plt.show()
