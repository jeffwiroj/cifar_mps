import torch.nn as nn
from torchvision.models import resnet18
from .resnet import ResNet


def get_models(model_name: str = "resnet18"):
    if model_name == "resnet18":

        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
    elif model_name == "smol_resnet":
        arch = [(1, 64, 64), (1, 64, 128), (1, 128, 256), (1, 256, 512)]
        model = ResNet(arch=arch, num_classes=10)
    else:
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
        print(f"Model {model_name} is not supported! Using Resnet18")
    return model
