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
import torchvision
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
#print(torchvision.__version__)

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None, is_train=True):
        self.data_folder = data_folder
        self.transform = transform
        self.data = []
        self.labels = []  # Use an empty label list, but it won't be used for test data
        self.is_train = is_train

        if self.is_train:
            # Iterate through all PNG image files in the data folder
            for img_filename in os.listdir(self.data_folder):
                if img_filename.endswith('.png'):
                    # Parse the filename to get class ID (i) and image number (j)
                    parts = img_filename.split('_')
                    if len(parts) == 2:
                        class_id = int(parts[0])
                        img_number = int(parts[1].split('.')[0])  # Remove file extension

                        img_path = os.path.join(self.data_folder, img_filename)
                        img = Image.open(img_path)
                        img_data = img.copy()  # Create a copy of the image to keep it open
                        img.close()  # Close the original image file
                        self.data.append(img_data)
                        self.labels.append(class_id)
        else:
            # If it's test data, only read PNG images
            for img_filename in os.listdir(self.data_folder):
                if img_filename.endswith('.png'):
                    img_path = os.path.join(self.data_folder, img_filename)
                    img = Image.open(img_path)
                    img_data = img.copy()  # Create a copy of the image to keep it open
                    img.close()  # Close the original image file
                    self.data.append(img_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
# Path to the data folders and figure folders
train_data_folder = './hw1_data/p1_data/train_50'
validation_data_folder = './hw1_data/p1_data/val_50'
figure_folder = './hw1_fig/'

# Define data preprocessing transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.RandomResizedCrop((224, 224), scale=(0.6, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

# Create custom datasets for both train and test data
train_dataset = CustomDataset(train_data_folder, transform=train_transform, is_train=True)
validation_dataset = CustomDataset(validation_data_folder, transform=validation_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)

# Initialize the custom classifier
num_class = 50 
epochs = 30
lr = 5e-5

# Define a custom classifier
class CustomEfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetV2Classifier, self).__init__()
        # Load the EfficientNetV2-S model
        self.effnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        #num_features = self.effnet_v2_s.fc.in_features
        # Replace the fully connected layer for your classification task
        self.effnet_v2_s.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.effnet_v2_s(x)
        return x


classifier = CustomEfficientNetV2Classifier(num_class).to(device)
#torchsummary.summary(classifier, input_size=(3, 224, 224))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=lr)

# Lists to record training loss, validation loss, training accuracy, validation accuracy, and PCA visualization results
train_loss_record = []
validation_loss_record = []
train_accuracy_record = []
validation_accuracy_record = []
best_val_accuracy = 0.0  # Initialize the best validation accuracy
pca_results = []
 
# Training loop (you can modify this for your specific training process)
for epoch in range(epochs):
    classifier.train()  # Set the model to training mode
    train_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    for images, labels in progress_bar:
        optimizer.zero_grad()  # Zero the gradients
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
        # Calculate training accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Calculate average training loss and training accuracy, and record them
    avg_train_loss = train_loss / len(train_dataloader)
    train_loss_record.append(avg_train_loss)
    train_accuracy = 100 * correct / total
    train_accuracy_record.append(train_accuracy)
    print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validation loop to evaluate the model on the validation set
    classifier.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Calculate average validation loss and record it
    avg_val_loss = val_loss / len(validation_dataloader)
    validation_loss_record.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    validation_accuracy_record.append(val_accuracy)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Check if the current model has the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        # Save the trained model with the best validation accuracy
        checkpoint_path = os.path.join('./model_checkpoint', f'P1_B_best_efficientnetv2_model_epoch_{epoch}.pth')
        torch.save(classifier.state_dict(), checkpoint_path)
    if (epoch+1)%10 == 0:
        checkpoint_path = os.path.join('./model_checkpoint', f'P1_B_efficientnetv2_model_epoch_{epoch}.pth')
        torch.save(classifier.state_dict(), checkpoint_path)

print(f"Best Validation Accuracy: {best_val_accuracy}")

# Plot and save the training and validation accuracy curves
plt.figure(figsize=(12, 6))
plt.plot(train_accuracy_record, label='Training Accuracy', marker='o')
plt.plot(validation_accuracy_record, label='Validation Accuracy', marker='o')
plt.title('p1_B_Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plot_filename = os.path.join(figure_folder, 'p1_B_accuracy_curves_final.png')
plt.savefig(plot_filename)

# Plot and save the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(train_loss_record, label='Training Loss', marker='o')
plt.plot(validation_loss_record, label='Validation Loss', marker='o')
plt.title('p1_B_Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plot_filename = os.path.join(figure_folder, 'p1_B_loss_curves_final.png')
plt.savefig(plot_filename)