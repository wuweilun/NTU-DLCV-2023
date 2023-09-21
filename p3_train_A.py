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

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import mean_iou_evaluate
import glob

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None, is_train=True):
        self.data_folder = data_folder
        self.transform = transform
        self.filename = sorted(glob.glob(os.path.join(data_folder, "*.jpg")))
        #self.filenames = [file for file in filepath]
        self.labels = []  # Use an empty label list, but it won't be used for test data
        self.is_train = is_train

        if self.is_train:
            self.labels = mean_iou_evaluate.read_masks(data_folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image
        
# Define a custom classifier
class VGG_FCN32(nn.Module):
    def __init__(self, num_classes):
        super(VGG_FCN32, self).__init__()

        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Extract the feature extraction part of VGG16 (convolutional layers)
        self.features = self.vgg16.features
        self.features[0].padding = (100, 100)
        
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score = nn.Conv2d(4096, num_classes, 1)
        # Define transpose convolutional layer for upsampling
        
        self.upsample32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32,
                                          padding=16, bias=False)
  

        #self.upsample32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)

        # Initialize the weights and biases of fc6 to match VGG16's corresponding part
        if hasattr(self.fc6, 'weight') and hasattr(self.fc6, 'bias'):
            vgg16_fc6_weight = self.vgg16.classifier[0].weight.data.view(self.fc6.weight.size())
            vgg16_fc6_bias = self.vgg16.classifier[0].bias.data.view(self.fc6.bias.size())
            self.fc6.weight.data.copy_(vgg16_fc6_weight)
            self.fc6.bias.data.copy_(vgg16_fc6_bias)

    def forward(self, x):
        x = self.upscore(x)
        x = self.features(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.score(x)
        
        # upsample to get segmentation prediction
        x = self.upsample32(x)
        return x

# Create an instance of the FCN32s model
num_classes = 7
vgg_fcn32 = VGG_FCN32(num_classes)

