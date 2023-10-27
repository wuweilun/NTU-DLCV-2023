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
from DANN_model import DANNModel

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

# Path to the data folders and figure folders
source_folder = './hw2_data/digits/mnistm/data'
train_source_csv_file = './hw2_data/digits/mnistm/train.csv'

target_folder = './hw2_data/digits/svhn/data'
train_target_csv_file = './hw2_data/digits/svhn/train.csv'
valid_target_csv_file = './hw2_data/digits/svhn/val.csv'

figure_folder = './hw2_fig/p3_train/'

# Create custom datasets for both train and test data
train_source_dataset = CustomDataset(source_folder, train_source_csv_file, transform=train_transform, is_train=True)
train_target_dataset = CustomDataset(target_folder, train_target_csv_file, transform=train_transform, is_train=True)
valid_target_dataset = CustomDataset(target_folder, valid_target_csv_file, transform=valid_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 2048
train_source_dataloader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=False)
train_target_dataloader = DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=False)
valid_target_dataloader = DataLoader(valid_target_dataset, batch_size=batch_size*2, shuffle=True, num_workers=8,  pin_memory=False)

# print(len(train_source_dataset), len(train_target_dataset))
# print(len(train_source_dataloader), len(train_target_dataloader))

# DANN model and settings
model = DANNModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
label_loss = nn.CrossEntropyLoss()
domain_loss = nn.BCEWithLogitsLoss()

epoch = 0
epochs = 100
num_class = 10
best_val_accuracy = 0.41

checkpoint = ''
checkpoint_name = ''
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)

if checkpoint is True:
    # load checkpoint
    checkpoint_info = torch.load(checkpoint_path)
    epoch = checkpoint_info['epoch'] + 1
    model.load_state_dict(checkpoint_info['model_state_dict'])
    optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])

total_iterations = max(len(train_source_dataloader), len(train_target_dataloader))

# Use itertools.cycle to create infinite data
source_iter = itertools.cycle(train_source_dataloader)
target_iter = itertools.cycle(train_target_dataloader)

# Training loop
while epoch < epochs:
    model.train()  # Set the model to training mode
    train_loss = 0.0
    train_label_loss = 0.0
    train_domain_loss = 0.0

    domain_total = 0
    domain_correct = 0
    source_label_total = 0
    source_label_correct = 0
    i = 0
    
    for _ in tqdm(range(total_iterations), desc=f"Epoch {epoch + 1}/{epochs}"):
        p = float(i + epoch * total_iterations) / epochs / total_iterations
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        optimizer.zero_grad()
        source_images, source_labels = next(source_iter)
        target_images, _ = next(target_iter)
        #source_images = torch.cat((source_images, source_images, source_images), 0)
        #source_labels = torch.cat((source_labels, source_labels, source_labels), 0)
        source_images, source_labels = source_images.to(device), source_labels.to(device)  # Move data to GPU
        target_images = target_images.to(device) # Move data to GPU
        
        # Use source domain data to train label classsifier 
        class_output, source_domain_output = model(input_data=source_images, alpha=alpha)
        source_label_loss = label_loss(class_output, source_labels)
        
        _, class_predicted = class_output.max(1)
        source_label_total += source_labels.size(0)
        source_label_correct += class_predicted.eq(source_labels).sum().item()
        
        # Use target domain data/target data to train domain classifier
        zeros = torch.zeros(source_images.shape[0]).float()
        ones = torch.ones(target_images.shape[0]).float()
        domain_labels = torch.cat((zeros, ones), 0).to(device)
        _, target_domain_output = model(input_data=target_images, alpha=alpha)
        
        domain_output = torch.cat((source_domain_output, target_domain_output), 0)
        domain_labels = domain_labels.view(-1, 1)
        
        total_domain_loss = domain_loss(domain_output, domain_labels)  
        
        predicted = torch.sigmoid(domain_output)
        predicted = torch.round(predicted)
        
        #_, predicted = domain_output.max(1)
        domain_total += domain_labels.size(0)
        domain_correct += predicted.eq(domain_labels).sum().item()

        # Error and backward
        loss = source_label_loss + total_domain_loss
        loss.backward()
        optimizer.step()
        
        train_loss += loss
        train_label_loss += source_label_loss
        train_domain_loss += total_domain_loss
        i += 1
        
    # Calculate average training loss
    avg_train_loss = train_loss / total_iterations
    avg_train_label_loss = train_label_loss / total_iterations
    avg_train_domain_loss = train_domain_loss / total_iterations
    print(f"[Train Loss] Total: {avg_train_loss:.4f}, Label : {avg_train_label_loss:.4f}, Source Domain: {avg_train_domain_loss:.4f}, alpha: {alpha}")

    label_train_accuracy = 100 * source_label_correct / source_label_total
    domain_train_accuracy = 100 * domain_correct / domain_total
    print(f"[Train Accuracy] Label: {label_train_accuracy:.2f}%, Domain: {domain_train_accuracy:.2f}%")
    
    # Validation loop to evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in valid_target_dataloader:
            labels = labels.long()

            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(input_data=images, alpha=alpha)
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
        checkpoint_path = os.path.join('./model_checkpoint', f'best_DANN_SVHN_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    elif (epoch+1)%10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'DANN_SVHN_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    epoch+=1