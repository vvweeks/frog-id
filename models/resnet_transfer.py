"""
models/resnet_transfer.py - ResNet18 ImageNet transfer learning model
for the mel-spectrogram pipeline.
"""
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES


def get_frog_model(dropout_p=0.4):
    print("--- Loading Pre-Trained ResNet-18 ---")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(num_ftrs, NUM_CLASSES),
    )
    return model


def freeze_bn_stats(model):
    """requires_grad=False stops weight updates on frozen layers, but
    NOT BatchNorm running_mean/running_var drift during model.train()
    forward passes. Call every epoch, right after model.train()."""
    frozen_submodules = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]
    for submodule in frozen_submodules:
        for m in submodule.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
