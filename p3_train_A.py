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
#import torchsummary 

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import mean_iou_evaluate
import glob
import random

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
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=True)

# Initialize the custom model
num_class = 7
epochs = 50
lr = 1e-4
checkpoint = True
checkpoint_name = 'P3_A_vggfcn32_model_epoch_19.pth'
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)

# Define a custom model
class VGG_FCN32(nn.Module):
    def __init__(self, num_classes):
        super(VGG_FCN32, self).__init__()

        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Extract the feature extraction part of VGG16 (convolutional layers)
        self.features = self.vgg16.features
        self.features[0].padding = (100, 100)
        
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score = nn.Conv2d(4096, num_classes, 1)
        # Define transpose convolutional layer for upsampling
        
        self.upsample32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32,
                                          padding=16, bias=False)
  

        #self.upsample32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)


    def forward(self, x):
        x = self.features(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.score(x)
        
        # upsample to get segmentation prediction
        x = self.upsample32(x)
        return x
    
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

# Create an instance of the FCN32s model
model = VGG_FCN32(num_class).to(device)
#torchsummary.summary(model, input_size=(3, 512, 512))

# Define loss function and optimizer
criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Lists to record training loss, validation loss, training mIOU, validation mIOU, and PCA visualization results
train_loss_record = []
validation_loss_record = []
validation_miou_record = []
best_val_miou = 0.69  # Initialize the best validation mIOU
best_val_loss = 100.0
epoch = 0

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
    num_batches = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    
    for images, labels in progress_bar:
        optimizer.zero_grad()  # Zero the gradients
        labels = labels.long()

        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average training loss and record
    avg_train_loss = train_loss / len(train_dataloader)
    train_loss_record.append(avg_train_loss)
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
            outputs = model(images)
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
        }
        # Save the trained model with the best validation mIOU
        checkpoint_path = os.path.join('./model_checkpoint', f'P3_A_best_vggfcn32_model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    if (epoch+1)%5 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join('./model_checkpoint', f'P3_A_vggfcn32_model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)    
    epoch+=1   
print(f"Best Validation loss: {best_val_loss}, Best Validation mIOU: {best_val_miou:.4f}")


# Plot and save the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(train_loss_record, label='Training Loss', marker='o')
plt.plot(validation_loss_record, label='Validation Loss', marker='o')
plt.title('P3_A_Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('p3_A_loss_curves.png')

# Plot and save the training and validation mIOU curves
plt.figure(figsize=(12, 6))
plt.plot(validation_miou_record, label='Validation mIOU', marker='o')
plt.title('p3_A_TValidation mIOU')
plt.xlabel('Epoch')
plt.ylabel('mIOU')
plt.legend()
plt.grid(True)
plt.savefig('p3_A_miou_curves.png')
