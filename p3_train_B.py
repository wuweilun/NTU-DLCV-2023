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
import torchsummary 
from torch.optim import lr_scheduler

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import mean_iou_evaluate
import random
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
        action = random.choice(['dummy','dummy2', 'hflip', 'vflip'])
            
        if self.is_train:
            label = Image.fromarray(self.labels[idx])
            if action == 'hflip':
                image = transforms.functional.hflip(image)
                label = transforms.functional.hflip(label)
            elif action == 'vflip':
                image = transforms.functional.vflip(image)
                label = transforms.functional.vflip(label)
            
            if self.transform:
                image = self.transform(image)
            label = np.array(label)
            return image, label
        else:
            if self.transform:
                image = self.transform(image)
            return image

# Define data preprocessing transformations
train_transform = transforms.Compose([
    transforms.ToTensor(),           # Convert images to Tensors
    #transforms.RandomResizedCrop((512, 512), scale=(0.6, 1)),
    #transforms.RandomHorizontalFlip(),
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

# Initialize the custom model
num_class = 7
epochs = 100
lr = 0.1
checkpoint = True
checkpoint_name = 'P3_B_best_deeplabv3_resnet50_model_epoch_5.pth'
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=6)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        #print('class #%d : %1.5f'%(i, iou))
    #print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

# Create an instance of the deeplabv3_resnet50 model
model = models.segmentation.deeplabv3_resnet50(num_classes = num_class, weight='DEFAULT').to(device)
#torchsummary.summary(model, input_size=(3, 512, 512))

# Define loss function and optimizer
criterion = FocalLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Lists to record training loss, validation loss results
train_loss_record = []
validation_loss_record = []
train_miou_record = []
validation_miou_record = []
best_val_miou = 0.0  # Initialize the best validation mIOU
best_val_loss = 100.0
epoch = 0

if checkpoint is True:
    # load checkpoint
    checkpoint_info = torch.load(checkpoint_path)
    epoch = checkpoint_info['epoch'] + 1
    model.load_state_dict(checkpoint_info['model_state_dict'])
    optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint_info['scheduler_state_dict'])
# Training loop
while epoch < epochs:
    model.train()  # Set the model to training mode
    train_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    labels_list = []
    pred_list = []
    
    for images, labels in progress_bar:
        optimizer.zero_grad()  # Zero the gradients
        labels = labels.long()

        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        # _, predicted = outputs.max(1)
        #labels_list.append(labels.detach().cpu().numpy().astype(np.int64))
        #pred_list.append(predicted.detach().cpu().numpy().astype(np.int64))
        
    scheduler.step()
    # Calculate average training loss, and record them
    avg_train_loss = train_loss / len(train_dataloader)
    #train_miou = mean_iou_score(np.concatenate(pred_list, axis=0), np.concatenate(labels_list, axis=0))
    train_loss_record.append(avg_train_loss)
    #train_miou_record.append(train_miou)
    #print(f"Training Loss: {avg_train_loss:.4f}, Training mIOU: {train_miou:.4f}")
    print(f"Training Loss: {avg_train_loss:.4f}")

    # Validation loop to evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    num_batches = 0
    labels_list = []
    pred_list = []
    
    with torch.no_grad():
        for images, labels in validation_dataloader:
            labels = labels.long()

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            labels_list.append(labels.detach().cpu().numpy().astype(np.int64))
            pred_list.append(predicted.detach().cpu().numpy().astype(np.int64))
            
    # Calculate average validation loss
    avg_val_loss = val_loss / len(validation_dataloader)
    val_miou = mean_iou_score(np.concatenate(pred_list, axis=0), np.concatenate(labels_list, axis=0))
    validation_loss_record.append(avg_val_loss)
    validation_miou_record.append(val_miou)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation mIOU: {val_miou:.4f}")

    # Check if the current model has the best validation mIOU
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        # Save the trained model with the best validation mIOU
        checkpoint_path = os.path.join('./model_checkpoint', f'P3_B_best_deeplabv3_resnet50_model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    if (epoch+1)%10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'P3_B_deeplabv3_resnet50_model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    epoch+=1
print(f"Best Validation loss: {best_val_loss}, Best Validation mIOU: {best_val_miou:.4f}")


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

# Plot and save the validation mIOU curves
plt.figure(figsize=(12, 6))
plt.plot(validation_miou_record, label='Validation mIOU', marker='o')
plt.title('p3_B_Validation mIOU')
plt.xlabel('Epoch')
plt.ylabel('mIOU')
plt.legend()
plt.grid(True)
plt.savefig('p3_B_miou_curves.png')