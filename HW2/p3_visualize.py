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
from PIL import Image
from tqdm import tqdm

import pandas as pd
import sys
import itertools
from DANN_model import DANNModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
        else:
            self.filenames = self.labels_df['filename'].tolist()
            self.ids = self.labels_df['id'].tolist()
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
            id = self.ids[idx]
            filename = self.filenames[idx]
            return image, id, filename

valid_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),     
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
    #                      std=[0.229, 0.224, 0.225])
])


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the data folders and figure folders
mnistm_folder = './hw2_data/digits/mnistm/data'
mnistm_csv_file = './hw2_data/digits/mnistm/val.csv'

svhn_folder = './hw2_data/digits/svhn/data'
svhn_csv_file = './hw2_data/digits/svhn/val.csv'

usps_folder = './hw2_data/digits/usps/data'
usps_csv_file = './hw2_data/digits/usps/val.csv'

figure_folder = './hw2_fig/p3_report/'

# Create custom datasets for both train and test data
mnistm_dataset = CustomDataset(mnistm_folder, mnistm_csv_file, transform=valid_transform, is_train=True)
svhn_dataset = CustomDataset(svhn_folder, svhn_csv_file, transform=valid_transform, is_train=True)
usps_dataset = CustomDataset(usps_folder, usps_csv_file, transform=valid_transform, is_train=True)

# Use DataLoaders to load the data
batch_size = 1024
mnistm_dataloader = DataLoader(mnistm_dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=False)
svhn_dataloader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=False)
usps_dataloader = DataLoader(usps_dataset, batch_size=batch_size, shuffle=True, num_workers=2,  pin_memory=False)

def get_activation(activation):
    def hook(model, input, output):
        activation.append(output.detach())
    return hook

# DANN model and settings
model = DANNModel().to(device)

target_list = ['usps', 'svhn']
model_chekpoint_list = ['p3_USPS/best_DANN_USPS_83.pth', 'p3_SVHN/best_DANN_SVHN_88.pth']

for target, checkpoint_name in zip(target_list, model_chekpoint_list):

    checkpoint_path = os.path.join('./model_checkpoint/', checkpoint_name)
    checkpoint_info = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_info['model_state_dict'])
        
    model.eval()

    # Perform TSNE visualization
    tsne_labels = []    # Initialize list to store t-SNE labels
    tsne_domains = [] 
    features = []
    model.feature_extractor.register_forward_hook(get_activation(features))
    if target == 'usps':   
        dataloader = usps_dataloader
    else:
        dataloader = svhn_dataloader
        
    with torch.no_grad():
        for images, labels in mnistm_dataloader:
            images, labels = images.to(device), labels.to(device)
            # Extract features from the CNN layer
            _, _ = model(images, 0)
            tsne_labels.extend(labels.cpu().numpy())
            tsne_domains.extend(np.zeros(len(labels)))
            
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # Extract features from the CNN layer
            _, _ = model(images, 0)
            tsne_labels.extend(labels.cpu().numpy())
            tsne_domains.extend(np.ones(len(labels)))

    tsne_labels = np.array(tsne_labels)
    tsne_domains = np.array(tsne_domains)
    for i in range(len(features)):
        features[i] = features[i].cpu()

    #print(features[0])
    features = torch.cat([batch_output.view(batch_output.size(0), -1) for batch_output in features], dim=0)

    # Perform t-SNE on the features
    tsne = TSNE(n_components=2)  
    tsne_result = tsne.fit_transform(features)

    # Create a scatter plot for t-SNE visualization
    plt.figure(figsize=(10, 8))
    for class_id in range(10):
        plt.scatter(tsne_result[tsne_labels==class_id, 0], 
                    tsne_result[tsne_labels==class_id, 1], 
                    label=class_id, alpha=0.5
        )
    plt.legend(loc='upper right')
    plt.title(f't-SNE Visualization MNISTM/{target} class)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # Save the t-SNE visualization plot to the "hw1_fig" directory
    tsne_plot_filename = os.path.join(figure_folder, f'tsne_MNISTM_{target}_class.png')
    plt.savefig(tsne_plot_filename)
    plt.close()

    plt.figure(figsize=(10, 8))
    for domain in range(2):
        plt.scatter(tsne_result[tsne_domains==domain, 0], 
                    tsne_result[tsne_domains==domain, 1], 
                    label=domain, alpha=0.2
        )
        
    plt.legend(loc='upper right')
    plt.title(f't-SNE Visualization MNISTM/{target} domain)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # Save the t-SNE visualization plot to the "hw1_fig" directory
    tsne_plot_filename = os.path.join(figure_folder, f'tsne_MNISTM_{target}_domain.png')
    plt.savefig(tsne_plot_filename)
    plt.close()
