import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
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
        self.filenames = sorted(glob.glob(os.path.join(data_folder, "*.jpg")))
        #self.filenames = [file for file in filepath]
        self.labels = []  # Use an empty label list, but it won't be used for test data
        self.is_train = is_train

        if self.is_train:
            self.labels = mean_iou_evaluate.read_masks(data_folder)
        #print(self.labels.shape)

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

# Define data preprocessing transformations
train_transform = transforms.Compose([
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.RandomResizedCrop((512, 512), scale=(0.6, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
# Path to the data folders and figure folders
train_data_folder = './hw1_data/p3_data/train'
validation_data_folder = './hw1_data/p3_data/validation'
figure_folder = './hw1_fig/'

# Create custom datasets for both train and test data
train_dataset = CustomDataset(train_data_folder, transform=train_transform, is_train=True)
validation_dataset = CustomDataset(validation_data_folder, transform=validation_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)

# Initialize the custom classifier
num_class = 7
epochs = 60
lr = 1e-4

# Create an instance of the deeplabv3_resnet50 model
classifier = models.segmentation.deeplabv3_resnet50(num_classes = num_class, weight='DEFAULT').to(device)
#torchsummary.summary(classifier, input_size=(3, 512, 512))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=lr)

# Lists to record training loss, validation loss results
train_loss_record = []
validation_loss_record = []
best_val_loss = 100.0
# Training loop
for epoch in range(epochs):
    classifier.train()  # Set the model to training mode
    train_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

    for images, labels in progress_bar:
        optimizer.zero_grad()  # Zero the gradients
        labels = labels.long()

        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = classifier(images)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Calculate average training loss, and record them
    avg_train_loss = train_loss / len(train_dataloader)
    train_loss_record.append(avg_train_loss)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # Validation loop to evaluate the model on the validation set
    classifier.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            labels = labels.long()

            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)['out']
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(validation_dataloader)
    validation_loss_record.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Check if the current model has the best validation mIOU
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the trained model with the best validation mIOU
        model_checkpoint = os.path.join('./model_checkpoint/', f'P3_B_best_deeplabv3_resnet50_model_epoch_{epoch}.pth')
        torch.save(classifier.state_dict(), model_checkpoint)
    if (epoch+1)%20 == 0:
        model_checkpoint = os.path.join('./model_checkpoint/', f'P3_B_deeplabv3_resnet50_model_epoch_{epoch}.pth')
        torch.save(classifier.state_dict(), model_checkpoint)

print(f"Best Validation loss: {best_val_loss}")

# Plot and save the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(train_loss_record, label='Training Loss', marker='o')
plt.plot(validation_loss_record, label='Validation Loss', marker='o')
plt.title('P3_B_Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('p3_B_loss_curves.png')
