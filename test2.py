import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchsummary 
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_class = 50
# Define a custom classifier
class CustomEfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetV2Classifier, self).__init__()
        # Load the EfficientNetV2-S model
        self.effnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        #num_features = self.effnet_v2_s.fc.in_features
        # Replace the fully connected layer for your classification task
        self.effnet_v2_s.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.effnet_v2_s(x)
        return x
# Define a custom classifier
class CustomResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)  # Load ResNet-18 without pre-trained weights
        num_features = self.resnet.fc.in_features  # Get the number of features in the fully connected layer
        self.resnet.fc = nn.Linear(num_features, num_classes)  # Replace the fully connected layer
        #print(num_features)
    def forward(self, x):
        x = self.resnet(x)
        return x

import torch
import torch.nn as nn

# Define the basic convolutional block (Convolutional Block)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output channels are not the same, an additional convolution layer is used for matching.
        self.downsample = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Define the ResNet-18 model
class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out



classifier = CustomEfficientNetV2Classifier(num_class).to(device)
torchsummary.summary(classifier, input_size=(3, 224, 224))

classifier2 = CustomResNetClassifier(num_class).to(device)
torchsummary.summary(classifier2, input_size=(3, 224, 224))

classifier3 = ResNet18(BasicBlock, [2, 2, 2, 2],num_classes=num_class).to(device)
torchsummary.summary(classifier3, input_size=(3, 224, 224))