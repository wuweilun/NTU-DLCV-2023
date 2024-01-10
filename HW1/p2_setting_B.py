import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
#import torchsummary 
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import glob
import random
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data_folder, csv_file, transform=None, is_train=True):
        self.data_folder = data_folder
        self.transform = transform
        self.is_train = is_train

        # Load labels from the CSV file
        self.labels_df = pd.read_csv(csv_file)

        # If it's a training set, load label data
        if self.is_train:
            self.filenames = self.labels_df['filename'].tolist()
            self.labels = self.labels_df['label'].tolist()
        else:
            self.filenames = sorted(glob.glob(os.path.join(data_folder, "*.jpg")))
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
            return image
        
image_size = 128
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),  
    RandomApply(
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        p = 0.3
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    RandomApply(
        transforms.GaussianBlur((3, 3), (1.0, 2.0)),
        p = 0.2
    ),
    transforms.RandomResizedCrop((image_size, image_size)),
    transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])),
])

validation_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
# Path to the data folders and figure folders
train_data_folder = './hw1_data/p2_data/office/train'
validation_data_folder = './hw1_data/p2_data/office/val'
figure_folder = './hw1_fig/'
train_csv_path = './hw1_data/p2_data/office/train.csv'
validation_csv_path = './hw1_data/p2_data/office/val.csv'
# Create custom datasets for both train and test data
train_dataset = CustomDataset(train_data_folder, train_csv_path, transform=train_transform, is_train=True)
validation_dataset = CustomDataset(validation_data_folder, validation_csv_path, transform=validation_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)

# Initialize the custom model
num_class = 65
epochs = 100
lr = 0.001
checkpoint = False
checkpoint_name = 'P2_setting_B_epoch_39.pth'
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)
backbone_name = 'pretrain_model_SL.pt'
backbone_path = os.path.join('./hw1_data/p2_data/', backbone_name)

# Create model
model = models.resnet50(weights=None)
model_checkpoint = torch.load(backbone_path)
# byol_model_state_dict = model_checkpoint['model_state_dict']
# resnet_state_dict = {}
# for key, value in byol_model_state_dict.items():
#     #print(key)
#     if key.startswith('net'):
#         new_key = key[len('net.'):]
#         resnet_state_dict[new_key] = value

model.load_state_dict(model_checkpoint)
#model.load_state_dict(model_checkpoint['model_state_dict'].online_encoder)
model.fc = nn.Linear(2048, num_class)
model.to(device)
#torchsummary.summary(model, input_size=(image_size, image_size,3 ))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Lists to record training loss, validation loss results
train_loss_record = []
validation_loss_record = []
train_accuracy_record = []
validation_accuracy_record = []
best_val_accuracy = 49.5  # Initialize the best validation mIOU
best_val_loss = 1000.0
epoch = 0

if checkpoint is True:
    # load checkpoint
    checkpoint_info = torch.load(checkpoint_path)
    epoch = checkpoint_info['epoch'] + 1
    model.load_state_dict(checkpoint_info['model_state_dict'])
    optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint_info['scheduler_state_dict'])
# Training loop
while epoch < epochs:
    model.train()  # Set the model to training mode
    train_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    correct = 0
    total = 0

    for images, labels in progress_bar:
        optimizer.zero_grad()  # Zero the gradients
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # Calculate training accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    #scheduler.step()
    # Calculate average training loss, and record them
    avg_train_loss = train_loss / len(train_dataloader)
    train_loss_record.append(avg_train_loss)
    train_accuracy = 100 * correct / total
    train_accuracy_record.append(train_accuracy)
    print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validation loop to evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(validation_dataloader)
    validation_loss_record.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    validation_accuracy_record.append(val_accuracy)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Check if the current model has the best validation mIOU
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
        }
        # Save the trained model with the best validation mIOU
        checkpoint_path = os.path.join('./model_checkpoint', f'P2_setting_B_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    if (epoch+1)%10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'P2_setting_B_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    epoch+=1
print(f"Best Validation loss: {best_val_loss}, Best Validation accuracy: {best_val_accuracy:.4f}")


