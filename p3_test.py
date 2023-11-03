import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

import pandas as pd
import sys
from DANN_model import DANNModel, NoDomainModel
import glob

def select_target_domain_from_path(file_path):
    if 'svhn' in file_path:
        target_domain = 'svhn'
    else:
        target_domain = 'usps'
    
    return target_domain

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None, is_train=False):
        self.data_folder = data_folder
        self.transform = transform
        self.is_train = is_train
        self.filenames = sorted(glob.glob(os.path.join(data_folder, "*.png")))
        
        self.labels = []  # Use an empty label list, but it won't be used for test data
        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        filename = os.path.basename(self.filenames[idx]).split('/')[-1]
        if self.transform:
            image = self.transform(image)

        if self.is_train == False:
            return image, filename

valid_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),     
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
    #                      std=[0.229, 0.224, 0.225])
])

# Path to testing images in the target domain, path to your output prediction file
data_path = sys.argv[1]
output_csv_path = sys.argv[2]

target_domain = select_target_domain_from_path(data_path)
print(f'target: {target_domain}')

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create custom datasets for both train and test data
dataset = CustomDataset(data_path, transform=valid_transform, is_train=False)

# Use DataLoaders to load the data
batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=False)

# DANN model and settings
model = DANNModel().to(device)
model.eval()

if target_domain == 'svhn':
    checkpoint_path = os.path.join('./', 'best_DANN_SVHN_88.pth')
else:
    checkpoint_path = os.path.join('./', 'best_DANN_USPS_83.pth')
    
checkpoint_info = torch.load(checkpoint_path)
model.load_state_dict(checkpoint_info['model_state_dict'])    

predictions = []
with torch.no_grad():
    for images, filenames in dataloader:
        images = images.to(device) 
        output, _ = model(images, 0)

        _, pred = torch.max(output, 1)
        predictions.extend(zip(filenames, pred.cpu().numpy()))

# Save the predictions results in csv file
result_df = pd.DataFrame(predictions, columns=['image_name', 'label'])
result_df = result_df.sort_values(by=['image_name'])  
result_df.to_csv(output_csv_path, index=False)
