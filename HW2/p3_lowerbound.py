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
from DANN_model import NoDomainModel

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

# Define data preprocessing transformations
train_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),     
    #transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
    #                      std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),     
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
    #                      std=[0.229, 0.224, 0.225])
])


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source = sys.argv[1]
target = sys.argv[2]

# Path to the data folders and figure folders
source_folder = f'./hw2_data/digits/{source}/data'
train_source_csv_file = f'./hw2_data/digits/{source}/train.csv'

target_folder = f'./hw2_data/digits/{target}/data'
valid_target_csv_file = f'./hw2_data/digits/{target}/val.csv'

figure_folder = './hw2_fig/p3_train/'

# Create custom datasets for both train and test data
train_source_dataset = CustomDataset(source_folder, train_source_csv_file, transform=train_transform, is_train=True)
valid_target_dataset = CustomDataset(target_folder, valid_target_csv_file, transform=valid_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 1024
train_source_dataloader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, num_workers=4,  pin_memory=True)
valid_target_dataloader = DataLoader(valid_target_dataset, batch_size=batch_size, shuffle=True, num_workers=4,  pin_memory=True)

# DANN model and settings
model = NoDomainModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
label_loss = nn.CrossEntropyLoss()

epoch = 0
epochs = 100
num_class = 10
best_val_accuracy = 0.7816

checkpoint = True
checkpoint_name = 'best_lowerbound_usps_46.pth'
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)

if checkpoint is True:
    # load checkpoint
    checkpoint_info = torch.load(checkpoint_path)
    epoch = checkpoint_info['epoch'] + 1
    model.load_state_dict(checkpoint_info['model_state_dict'])
    optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])


# Training loop
while epoch < epochs:
    model.train()  # Set the model to training mode
    train_loss = 0.0
    train_label_loss = 0.0

    source_label_total = 0
    source_label_correct = 0
    i = 0
    
    progress_bar = tqdm(train_source_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    for source_images, source_labels in progress_bar:
        optimizer.zero_grad()

        source_images, source_labels = source_images.to(device), source_labels.to(device)  # Move data to GPU
        
        # Use source domain data to train label classsifier 
        class_output = model(input_data=source_images)
        source_label_loss = label_loss(class_output, source_labels)
        
        _, class_predicted = class_output.max(1)
        source_label_total += source_labels.size(0)
        source_label_correct += class_predicted.eq(source_labels).sum().item()
        
        # Error and backward
        loss = source_label_loss
        loss.backward()
        optimizer.step()
        
        train_loss += loss
        train_label_loss += source_label_loss
        i += 1
        
    # Calculate average training loss
    avg_train_loss = train_loss / len(train_source_dataloader)
    avg_train_label_loss = train_label_loss / len(train_source_dataloader)
    print(f"[Train Loss] Total: {avg_train_loss:.4f}, Label : {avg_train_label_loss:.4f}")

    label_train_accuracy = 100 * source_label_correct / source_label_total
    print(f"[Train Accuracy] Label: {label_train_accuracy:.2f}%")

    # Validation loop to evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in valid_target_dataloader:
            labels = labels.long()

            images, labels = images.to(device), labels.to(device)
            outputs = model(input_data=images)
            loss = label_loss(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Calculate average validation loss and record it
    avg_val_loss = val_loss / len(valid_target_dataloader)
    val_accuracy = 100 * correct / total
    print(f"[Validation] Target Label Loss: {avg_val_loss:.4f}, Target Label Accuracy: {val_accuracy:.2f}%")
    
    # Check if the current model has the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'best_lowerbound_{target}_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    epoch+=1
    
print(f"Best Validation Accuracy: {best_val_accuracy}%")
