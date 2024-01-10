# DDPM architecture: https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py

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
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

import pandas as pd
import sys
from unet_conditional import UNet_conditional

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

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_folder, self.filenames[idx]))
        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.labels[idx]
            return image, label

# Define data preprocessing transformations
train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),     
    #transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Path to the data folders and figure folders
train_data_folder = './hw2_data/digits/mnistm/data'
train_csv_file = './hw2_data/digits/mnistm/train.csv'
figure_folder = './hw2_fig/p1_train/'

# Create custom datasets for both train and test data
train_dataset = CustomDataset(train_data_folder, train_csv_file, transform=train_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=28, num_classes=10, c_in=3, c_out=3):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes).to(device)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample(self, labels, cfg_scale=3):
        model = self.model
        n = len(labels)
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps))):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x

ddpm = Diffusion(img_size=32)
optimizer = optim.AdamW(ddpm.model.parameters(), lr=5e-5) # lr=1e-4
mse = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()

epoch = 0
epochs = 1000
checkpoint = True
checkpoint_name = 'ddpm_879.pth'
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)

if checkpoint is True:
    # load checkpoint
    checkpoint_info = torch.load(checkpoint_path)
    epoch = checkpoint_info['epoch'] + 1
    ddpm.model.load_state_dict(checkpoint_info['model_state_dict'])
    optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])

# Training loop
while epoch < epochs:
    ddpm.model.train()  # Set the model to training mode
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    
    for images, labels in progress_bar:
        optimizer.zero_grad()  # Zero the gradients

        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        t = ddpm.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = ddpm.noise_images(images, t)
        if np.random.random() < 0.1:
            labels = None
        predicted_noise = ddpm.model(x_t, t, labels)
        loss = mse(noise, predicted_noise)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress_bar.set_postfix(loss=loss.item())
        train_loss += loss

        
    # Calculate average training loss
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Training Loss: {avg_train_loss:.4f}")

    if (epoch+1)%10 == 0:
        # Validation loop to evaluate the model 
        ddpm.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            labels = torch.arange(ddpm.num_classes).long().to(device)
            sampled_images = ddpm.sample(labels=labels)
            grid = make_grid(sampled_images, nrow=2)
            save_image(grid, figure_folder+f'sample_images_{epoch}.png')
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ddpm.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'ddpm_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    epoch+=1

