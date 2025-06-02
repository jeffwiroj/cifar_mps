import torch
from torchvision.transforms import v2, RandAugment


def get_transformation(transformation_type: str) -> v2.Compose:
    """
    Get data transformations for CIFAR datasets.

    Args:
        transformation_type: Type of transformation ('basic', 'augmented', 'test', etc.)

    Returns:
        Composed transformation pipeline

    Raises:
        ValueError: If transformation_type is not supported
    """

    # CIFAR-10 dataset statistics
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)

    transformations = {
        "basic": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        ),
        "random": v2.Compose(
            [
                v2.ToImage(),
                RandAugment(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        ),
        "test": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        ),
    }

    if transformation_type not in transformations:
        available = ", ".join(transformations.keys())
        raise ValueError(
            f"Unknown transformation_type '{transformation_type}'. "
            f"Available options: {available}"
        )

    return transformations[transformation_type]
