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
from byol_pytorch import BYOL

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.filenames = sorted(glob.glob(os.path.join(data_folder, "*.jpg")))
        self.labels = []  # Use an empty label list, but it won't be used for test data


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        if self.transform:
            image = self.transform(image)
        return image
    
train_transform = transforms.Compose([
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Resize(128),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
    #                      std=[0.229, 0.224, 0.225])
])

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the data folders and figure folders
train_data_folder = './hw1_data/p2_data/mini/train'
figure_folder = './hw1_fig/'

# Create custom datasets for both train and test data
train_dataset = CustomDataset(train_data_folder, transform=train_transform)

# Create a DataLoader
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Initialize a ResNet model without pretrained weights
resnet = models.resnet50(weights=None).to(device)
resnet.fc = nn.Linear(2048, 64) # trick, we know that total have 64 classes
model = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool',
    use_momentum = False,
)

# Initialize the custom model
lr = 0.0003
checkpoint = True
checkpoint_name = 'P2_best_byol_model_epoch_38.pth'
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define CosineAnnealingLR scheduler for later epochs
#cosine_annealing_epochs = 800
#cosine_annealing_scheduler = CosineAnnealingLR(optimizer, cosine_annealing_epochs)

epochs = 350
epoch = 0
best_train_loss = 0.3
train_loss_record = []

if checkpoint is True:
    # load checkpoint
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']+1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # if epoch < warmup_epochs:
    #     warmup_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # else:
    #cosine_annealing_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

while epoch <epochs:
    model.train()  # Set the model to training mode
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    for batch in progress_bar:
        images = batch.to(device)
        loss = model(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #model.update_moving_average()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_dataloader)
    train_loss_record.append(avg_train_loss)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # Update the learning rate using the warm-up scheduler for the initial epochs
    # if epoch < warmup_epochs:
    #     warmup_scheduler.step()
    # else:
        # Switch to CosineAnnealingLR scheduler
    #cosine_annealing_scheduler.step()

    # Check if the current model has the best train loss
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': cosine_annealing_scheduler.state_dict(),
            # 'scheduler_state_dict': warmup_scheduler.state_dict() if epoch < warmup_epochs else cosine_annealing_scheduler.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'P2_best_byol_model_setting4_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

    # Save model, optimizer, scheduler, and learning rate every save_every_n_epochs epochs
    if (epoch + 1) % 30 == 0:
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': cosine_annealing_scheduler.state_dict(),
            #'scheduler_state_dict': warmup_scheduler.state_dict() if epoch < warmup_epochs else cosine_annealing_scheduler.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'P2_byol_model_setting4_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

    epoch+=1
    

# # Plot and save the training and validation loss curves
# plt.figure(figsize=(12, 6))
# plt.plot(train_loss_record, label='Training Loss', marker='o')
# plt.title('P2_BYOL_Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig('p2_BYOL_loss_curves.png')
