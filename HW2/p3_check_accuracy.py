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
import itertools
from DANN_model import DANNModel, NoDomainModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data_folder, csv_file, transform=None, is_train=True):
        self.data_folder = data_folder
        self.transform = transform
        self.is_train = is_train
        
        # Load labels from the CSV file
        self.labels_df = pd.read_csv(csv_file)
        
        # If it's a training set, load label data
        if self.is_train:
            self.filenames = self.labels_df['image_name'].tolist()
            self.labels = self.labels_df['label'].tolist()
        #print(f"Number of images: {len(self.filenames)}")
        else:
            self.filenames = self.labels_df['filename'].tolist()
            self.ids = self.labels_df['id'].tolist()
            self.labels = []  # For test data, labels can be an empty list

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_folder, self.filenames[idx]))
        
        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            id = self.ids[idx]
            filename = self.filenames[idx]
            return image, id, filename

valid_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),     
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
    #                      std=[0.229, 0.224, 0.225])
])


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the data folders and figure folders
svhn_folder = './hw2_data/digits/svhn/data'
svhn_csv_file = './hw2_data/digits/svhn/val.csv'

usps_folder = './hw2_data/digits/usps/data'
usps_csv_file = './hw2_data/digits/usps/val.csv'

figure_folder = './hw2_fig/p3_report/'

# Create custom datasets for both train and test data
svhn_dataset = CustomDataset(svhn_folder, svhn_csv_file, transform=valid_transform, is_train=True)
usps_dataset = CustomDataset(usps_folder, usps_csv_file, transform=valid_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 1024
svhn_dataloader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=False)
usps_dataloader = DataLoader(usps_dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=False)

# DANN model and settings
DANN_model = DANNModel().to(device)
NoDomain_model = NoDomainModel().to(device)
DANN_model.eval()
NoDomain_model.eval()

target_list = ['usps', 'svhn']
model_chekpoint_list = [['p3_USPS/best_lowerbound_usps_90.pth', 'p3_USPS/best_DANN_USPS_83.pth', 'p3_USPS/best_upperbound_usps_61.pth'], 
                        ['p3_SVHN/best_lowerbound_svhn_13.pth', 'p3_SVHN/best_DANN_SVHN_88.pth', 'p3_SVHN/best_upperbound_svhn_94.pth']]
method_list = ['lower bound', 'DANN method', 'upper bound']
for target, checkpoint_name in zip(target_list, model_chekpoint_list):
    if target == 'usps':   
        dataloader = usps_dataloader
    else:
        dataloader = svhn_dataloader
        
    for method in method_list:
        if method == 'lower bound':
            checkpoint_path = os.path.join('./model_checkpoint/', checkpoint_name[0])
            model = NoDomain_model
        elif method =='DANN method':
            checkpoint_path = os.path.join('./model_checkpoint/', checkpoint_name[1])
            model = DANN_model
        else:
            checkpoint_path = os.path.join('./model_checkpoint/', checkpoint_name[2])
            model = NoDomain_model
            
        checkpoint_info = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_info['model_state_dict'])    
        correct = 0
        total = 0   
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                if method == 'DANN method':
                    output, _ = model(images, 0)
                else:
                    output = model(images)
                _, pred = torch.max(output, 1)
                correct += (pred == labels).detach().sum().item()
                total += len(pred)
        accuracy = 100 *correct/total
        print(f"{target} {method} accuracy: {accuracy:.2f}%, (correct/total = {correct}/{total})")

