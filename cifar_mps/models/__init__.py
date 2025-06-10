import torch.nn as nn
from torchvision.models import resnet18
from .resnet import ResNet


def get_models(model_name: str = "resnet18", **kwargs):
    if model_name == "resnet18":

        model = resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()
    elif model_name == "smol_resnet":
        arch = [(1, 64, 64), (1, 64, 128), (1, 128, 256), (1, 256, 256)]
        model = ResNet(arch=arch, num_classes=10)
    else:
        model = resnet18(weights=None, num_classes=10, **kwargs)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()
        print(f"Model {model_name} is not supported! Using Resnet18")
    return model
