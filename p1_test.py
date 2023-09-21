import argparse
import os
import sys

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
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None, is_train=False):
        self.data_folder = data_folder
        self.transform = transform
        self.data = []
        self.labels = []  # Use an empty label list, but it won't be used for test data
        self.is_train = is_train
        self.filenames = []
        
        if is_train == False:
            # If it's test data, only read PNG images
            for img_filename in os.listdir(self.data_folder):
                if img_filename.endswith('.png'):
                    img_path = os.path.join(self.data_folder, img_filename)
                    img = Image.open(img_path)
                    img_data = img.copy()  # Create a copy of the image to keep it open
                    img.close()  # Close the original image file
                    self.data.append(img_data)
                    self.filenames.append(img_filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        filename = self.filenames[idx]
        
        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image, filename

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the data folders and output csv file
test_data_folder = sys.argv[1]
output_csv_path = sys.argv[2]

# Define data preprocessing transformations

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

# Create custom datasets for test data
test_dataset = CustomDataset(test_data_folder, transform=test_transform, is_train=False)

# Use DataLoaders to load the data
batch_size = 64
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,  pin_memory=True)

# Initialize the custom classifier
num_class = 50 

# Define a custom classifier
class CustomEfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetV2Classifier, self).__init__()
        # Load the EfficientNetV2-S model
        self.effnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.effnet_v2_s.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.effnet_v2_s(x)
        return x


classifier = CustomEfficientNetV2Classifier(num_class).to(device)
classifier.load_state_dict(torch.load('P1_B_best_custom_efficientnetv2_model_best.pth'))  
classifier.eval()  

# Use classifier to predict class
predictions = []

with torch.no_grad():
    for images, filenames in test_dataloader:
        images = images.to(device)
        outputs = classifier(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(zip(filenames, predicted.cpu().numpy()))

# Save the predictions results in csv file
result_df = pd.DataFrame(predictions, columns=['filename', 'label'])
result_df = result_df.sort_values(by=['label', 'filename'])  
result_df.to_csv(output_csv_path, index=False)

