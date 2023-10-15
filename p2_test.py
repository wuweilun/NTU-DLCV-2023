import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from PIL import Image
import pandas as pd
import sys

class CustomDataset(Dataset):
    def __init__(self, data_folder, csv_file, transform=None, is_train=True):
        self.data_folder = data_folder
        self.transform = transform
        self.is_train = is_train
        
        # Load labels from the CSV file
        self.labels_df = pd.read_csv(csv_file)
        
        # If it's a training set, load label data
        if self.is_train:
            self.filenames = self.labels_df['filename'].tolist()
            self.labels = self.labels_df['label'].tolist()
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
        
image_size = 128

validation_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
# Path to the data folders and figure folders
validation_csv_file =sys.argv[1]
validation_data_folder = sys.argv[2]
output_path = sys.argv[3]
# Create custom datasets for both train and test data
validation_dataset = CustomDataset(validation_data_folder, validation_csv_file, transform=validation_transform, is_train=False)

# Use DataLoaders to load the data
batch_size = 256
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8,  pin_memory=True)

# Initialize the custom model
num_class = 65

model_checkpint_folder = './hw1_model_checkpoint'
checkpoint_name = 'P2_setting_C_epoch_86.pth'
checkpoint_path = os.path.join(model_checkpint_folder, checkpoint_name)
#backbone_name = 'P2_best_byol_model_epoch_38.pth'
#backbone_path = os.path.join('./model_checkpoint', backbone_name)

# Create model
model = models.resnet50(weights=None)
# model_checkpoint = torch.load(backbone_path)
# byol_model_state_dict = model_checkpoint['model_state_dict']
# resnet_state_dict = {}
# for key, value in byol_model_state_dict.items():
#     #print(key)
#     if key.startswith('net'):
#         new_key = key[len('net.'):]
#         resnet_state_dict[new_key] = value
        
# model.fc = nn.Linear(2048, 64)
# model.load_state_dict(resnet_state_dict)
checkpoint_info = torch.load(checkpoint_path)['model_state_dict']
model.fc = nn.Linear(2048, num_class)
model.load_state_dict(checkpoint_info)
model.to(device)
#torchsummary.summary(model, input_size=(image_size, image_size,3 ))

# Validation loop to evaluate the model on the validation set
model.eval()  # Set the model to evaluation mode

predictions = []
with torch.no_grad():
    for images, id, filenames in validation_dataloader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        predictions.extend(zip(id.cpu().numpy(), filenames, predicted.cpu().numpy()))

# Save the predictions results in csv file
result_df = pd.DataFrame(predictions, columns=['id', 'filename', 'label'])
result_df.to_csv(output_path, index=False)