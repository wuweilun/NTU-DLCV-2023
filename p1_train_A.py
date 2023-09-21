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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
#torch.backends.cudnn.enabled = False
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

# Path to the data folders and figure folders
train_data_folder = './hw1_data/p1_data/train_50'
validation_data_folder = './hw1_data/p1_data/val_50'
figure_folder = './hw1_fig/'

# Define data preprocessing transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),           # Convert images to Tensors
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
epochs = 50
lr = 1e-4

'''
# Define a custom classifier
class CustomResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)  # Load ResNet-18 without pre-trained weights
        num_features = self.resnet.fc.in_features  # Get the number of features in the fully connected layer
        self.resnet.fc = nn.Linear(num_features, num_classes)  # Replace the fully connected layer
        #print(num_features)
    def forward(self, x):
        x = self.resnet(x)
        return x
'''
# Define the basic convolutional block (Convolutional Block)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output channels are not the same, an additional convolution layer is used for matching.
        self.downsample = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Define the ResNet-18 model
class CustomResNetClassifier(nn.Module):
    def __init__(self, block, num_blocks, num_classes=50):
        super(CustomResNetClassifier, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

def get_activation(activation):
    def hook(model, input, output):
        activation.append(output.detach())
    return hook

classifier = CustomResNetClassifier(BasicBlock, [2, 2, 2, 2],num_classes=num_class).to(device)

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
        torch.save(classifier.state_dict(), 'P1_A_best_custom_resnet_model.pth')
        
    # Perform PCA & TSNE visualization
    if epoch % 10 == 0:  # adjust the frequency of PCA visualization
        classifier.eval()  # Set the model to evaluation mode for PCA
        pca_labels = []    # Initialize list to store PCA labels
        tsne_labels = []    # Initialize list to store t-SNE labels
        features = []
        classifier.layer4.register_forward_hook(get_activation(features))
        
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                # Extract features from the second last layer
                outputs = classifier(images)
                pca_labels.extend(labels.cpu().numpy())
                tsne_labels.extend(labels.cpu().numpy())

        pca_labels = np.array(pca_labels)
        tsne_labels = np.array(tsne_labels)
        for i in range(len(features)):
            features[i] = features[i].cpu()
        
        #print(pca_labels.shape)
        #print(features[0])
        features = torch.cat([batch_output.view(batch_output.size(0), -1) for batch_output in features], dim=0)
        # Perform PCA on the features
        pca = PCA(n_components=2)  
        pca_result = pca.fit_transform(features)

        # Perform t-SNE on the features
        tsne = TSNE(n_components=2)  
        tsne_result = tsne.fit_transform(features)

        # Create a scatter plot for PCA visualization
        plt.figure(figsize=(10, 8))
        for class_id in range(num_class):
            plt.scatter(pca_result[pca_labels==class_id, 0], 
                        pca_result[pca_labels==class_id, 1], 
                        alpha=0.5
            )

        plt.title(f'PCA Visualization (Epoch {epoch + 1})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        
        # Save the PCA visualization plot
        pca_plot_filename = os.path.join(figure_folder, f'./pca/pca_epoch_{epoch + 1}.png')
        plt.savefig(pca_plot_filename)
        plt.close()
        
        # Create a scatter plot for t-SNE visualization
        plt.figure(figsize=(10, 8))
        for class_id in range(num_class):
            plt.scatter(tsne_result[tsne_labels==class_id, 0], 
                        tsne_result[tsne_labels==class_id, 1], 
                        alpha=0.5
            )

        plt.title(f't-SNE Visualization (Epoch {epoch + 1})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)

        # Save the t-SNE visualization plot to the "hw1_fig" directory
        tsne_plot_filename = os.path.join(figure_folder, f'./tsne/tsne_epoch_{epoch + 1}.png')
        plt.savefig(tsne_plot_filename)
        plt.close()
        # Record the PCA results for later analysis or comparison
        pca_results.append((epoch + 1, pca_result, pca_labels))

        
print(f"Best Validation Accuracy: {best_val_accuracy}")

# Plot and save the training and validation accuracy curves
plt.figure(figsize=(12, 6))
plt.plot(train_accuracy_record, label='Training Accuracy', marker='o')
plt.plot(validation_accuracy_record, label='Validation Accuracy', marker='o')
plt.title('p1_A_Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plot_filename = os.path.join(figure_folder, 'p1_A_accuracy_curves.png')
plt.savefig(plot_filename)

# Plot and save the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(train_loss_record, label='Training Loss', marker='o')
plt.plot(validation_loss_record, label='Validation Loss', marker='o')
plt.title('p1_A_Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plot_filename = os.path.join(figure_folder, 'p1_A_loss_curves.png')
plt.savefig(plot_filename)