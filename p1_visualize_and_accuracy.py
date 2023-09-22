import pandas as pd
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
import sys
import os

ground_truth_df = pd.read_csv('./hw1_data/p1_data/val_gt.csv') 
predictions_df = pd.read_csv('./bash_output/p1_pred.csv')  

merged_df = pd.merge(ground_truth_df, predictions_df, left_on='image_id', right_on='filename', suffixes=('_truth', '_prediction'))

correct_predictions = (merged_df['label_truth'] == merged_df['label_prediction']).sum()

total_samples = len(merged_df)

accuracy = correct_predictions / total_samples

print(f'Accuracy: {accuracy * 100:.2f}%')

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
figure_folder = './hw1_fig/'
validation_data_folder = './hw1_data/p1_data/val_50'
# Define data preprocessing transformations

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

# Create custom datasets for test data
validation_dataset = CustomDataset(validation_data_folder, transform=test_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 64
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0,  pin_memory=True)

# Initialize the custom classifier
num_class = 50 

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
model_checkpoint_list = [f'P1_A_custom_resnet_model_epoch{epoch}' for epoch in [1, 11, 21, 31, 41]]
for model_name in model_checkpoint_list:
    classifier.load_state_dict(torch.load(f'{model_name}.pth'))  
    classifier.eval()  # Set the model to evaluation mode for PCA

    # Perform PCA & TSNE visualization
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

    plt.title(f'PCA Visualization: {model_name})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    # Save the PCA visualization plot
    pca_plot_filename = os.path.join(figure_folder, f'./pca/pca_{model_name}.png')
    plt.savefig(pca_plot_filename)
    plt.close()

    # Create a scatter plot for t-SNE visualization
    plt.figure(figsize=(10, 8))
    for class_id in range(num_class):
        plt.scatter(tsne_result[tsne_labels==class_id, 0], 
                    tsne_result[tsne_labels==class_id, 1], 
                    alpha=0.5
        )

    plt.title(f't-SNE Visualization {model_name})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # Save the t-SNE visualization plot to the "hw1_fig" directory
    tsne_plot_filename = os.path.join(figure_folder, f'./tsne/tsne_{model_name}.png')
    plt.savefig(tsne_plot_filename)
    plt.close()
