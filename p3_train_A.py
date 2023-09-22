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
epochs = 30
lr = 1e-4

# Define a custom classifier
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

        # # Initialize the weights and biases of fc6 to match VGG16's corresponding part
        # if hasattr(self.fc6, 'weight') and hasattr(self.fc6, 'bias'):
        #     vgg16_fc6_weight = self.vgg16.classifier[0].weight.data.view(self.fc6.weight.size())
        #     vgg16_fc6_bias = self.vgg16.classifier[0].bias.data.view(self.fc6.bias.size())
        #     self.fc6.weight.data.copy_(vgg16_fc6_weight)
        #    self.fc6.bias.data.copy_(vgg16_fc6_bias)

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

# Create an instance of the FCN32s model
classifier = VGG_FCN32(num_class).to(device)
torchsummary.summary(classifier, input_size=(3, 512, 512))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=lr)

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over multiple images
    Args:
        pred: Predicted labels (batch_size, width, height)
        labels: True labels (batch_size, width, height)
    '''
    batch_size = pred.shape[0]
    mean_iou = 0

    for i in range(batch_size):
        pred_single = pred[i]
        labels_single = labels[i]
        #print(pred_single.shape)
        #print(labels_single.shape)
        # Calculate IoU for each class
        iou_per_class = []
        for class_id in range(6):
            tp_fp = np.sum(pred_single == class_id)
            tp_fn = np.sum(labels_single == class_id)
            tp = np.sum((pred_single == class_id) * (labels_single == class_id))
            iou = tp / (tp_fp + tp_fn - tp)
            iou_per_class.append(iou)

        # Calculate mean IoU for this image
        mean_iou += np.mean(iou_per_class)

    # Calculate the overall mean IoU across all images
    mean_iou /= batch_size

    return mean_iou


# Lists to record training loss, validation loss, training mIOU, validation mIOU, and PCA visualization results
train_loss_record = []
validation_loss_record = []
train_miou_record = []
validation_miou_record = []
best_val_miou = 0.0  # Initialize the best validation mIOU
best_val_loss = 100.0
# Training loop 
for epoch in range(epochs):
    classifier.train()  # Set the model to training mode
    train_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    
    for images, labels in progress_bar:
        optimizer.zero_grad()  # Zero the gradients
        labels = labels.long()

        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # Calculate mIOU for this batch
        # _, predicted = outputs.max(1)
        # print(predicted.shape)
        # print(labels.shape)
        # iou = mean_iou_score(predicted, labels)  # You need to implement calculate_miou function
        # total_iou += iou
        # num_batches += 1
    
    # Calculate average training loss and training mIOU, and record them
    avg_train_loss = train_loss / len(train_dataloader)
    # avg_train_miou = total_iou / num_batches
    train_loss_record.append(avg_train_loss)
    # train_miou_record.append(avg_train_miou)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # Validation loop to evaluate the model on the validation set
    classifier.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    #total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            labels = labels.long()

            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # # Calculate mIOU for this batch
            # _, predicted = outputs.max(1)
            # iou = mean_iou_score(predicted, labels)  # You need to implement calculate_miou function
            # total_iou += iou
            # num_batches += 1
    
    # Calculate average validation loss and validation mIOU, and record them
    avg_val_loss = val_loss / len(validation_dataloader)
    #avg_val_miou = total_iou / num_batches
    validation_loss_record.append(avg_val_loss)
    #validation_miou_record.append(avg_val_miou)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Check if the current model has the best validation mIOU
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the trained model with the best validation mIOU
        torch.save(classifier.state_dict(), f'P3_A_best_vggfcn32_model_epoch_{epoch}.pth')
    if (epoch+1)%5 == 0:
        torch.save(classifier.state_dict(), f'P3_A_vggfcn32_model_epoch_{epoch}.pth')

print(f"Best Validation loss: {best_val_loss}")

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
# plt.figure(figsize=(12, 6))
# plt.plot(train_miou_record, label='Training mIOU', marker='o')
# plt.plot(validation_miou_record, label='Validation mIOU', marker='o')
# plt.title('p3_A_Training and Validation mIOU')
# plt.xlabel('Epoch')
# plt.ylabel('mIOU')
# plt.legend()
# plt.grid(True)
# plt.savefig('p3_A_miou_curves.png')
