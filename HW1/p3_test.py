import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from PIL import Image
import imageio

import glob
import sys

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land
        masks[i, mask == 1] = 4  # (Blue: 001) Water
        masks[i, mask == 7] = 5  # (White: 111) Barren land
        masks[i, mask == 0] = 6  # (Black: 000) Unknown

    return masks

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None, is_train=True):
        self.data_folder = data_folder
        self.transform = transform
        self.filenames = sorted(glob.glob(os.path.join(data_folder, "*.jpg")))
        #self.filenames = [file for file in filepath]
        self.labels = []  # Use an empty label list, but it won't be used for test data
        self.is_train = is_train


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        filename = os.path.basename(self.filenames[idx]).split('/')[-1]
        if self.transform:
            image = self.transform(image)

        if self.is_train == False:
            return image, filename

validation_transform = transforms.Compose([
    transforms.ToTensor(),           # Convert images to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                         std=[0.229, 0.224, 0.225])
])

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
# Path to the data folders and figure folders
validation_data_folder = sys.argv[1]
output_path = sys.argv[2]


# Create custom datasets for both train and test data
validation_dataset = CustomDataset(validation_data_folder, transform=validation_transform, is_train=False)

# Use DataLoaders to load the data
batch_size = 8
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8,  pin_memory=True)


# Initialize the custom model
num_class = 7

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
model_checkpint_folder = './hw1_model_checkpoint'
checkpoint_name = 'P3_B_best_deeplabv3_resnet50_model_epoch_86.pth'
#checkpoint_name = 'P3_B_deeplabv3_resnet50_model_epoch_9.pth'
checkpoint_path = os.path.join(model_checkpint_folder, checkpoint_name)

# Create an instance of the deeplabv3_resnet50 model
model = models.segmentation.deeplabv3_resnet50(num_classes = num_class, weight=None).to(device)
checkpoint_info = torch.load(checkpoint_path)['model_state_dict']
model.load_state_dict(checkpoint_info)
model.eval()  # Set the model to evaluation mode

predictions = []
with torch.no_grad():
    for images, filenames in validation_dataloader:
        images = images.to(device)
        outputs = model(images)['out']

        _, predicted = outputs.max(1)
        #print(predicted)
        #print(predicted.shape)
        predictions.extend(zip(filenames, predicted.cpu().numpy()))
        
color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0]}

for filename, pred in predictions:
    #print(filename.shape)
    #print(pred.shape)
    #print(filename)
    pred_image = np.zeros((512, 512, 3), dtype=np.uint8)
    pred_image[np.where(pred == 0)] = color[0]
    pred_image[np.where(pred == 1)] = color[1]
    pred_image[np.where(pred == 2)] = color[2]
    pred_image[np.where(pred == 3)] = color[3]
    pred_image[np.where(pred == 4)] = color[4]
    pred_image[np.where(pred == 5)] = color[5]
    pred_image[np.where(pred == 6)] = color[6]
    #print(pred_image)
    #print(pred_image.shape)
    filename = filename.replace('sat.jpg', 'mask.png')
    #print(filename)
    img = Image.fromarray(np.uint8(pred_image))
    #imageio.imwrite(os.path.join(output_path, filename), pred_image)
    img.save(os.path.join(output_path, filename))