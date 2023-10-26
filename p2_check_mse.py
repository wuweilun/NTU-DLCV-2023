import torch
import torch.nn as nn
import glob
import os
from PIL import Image
import numpy as np
import os

gt_data_folder = './hw2_data/face/GT/'
generated_data_folder = './hw2_fig/p2_mse/'
gt_filenames = sorted(glob.glob(os.path.join(gt_data_folder, "*.png")))
generated_filenames = sorted(glob.glob(os.path.join(generated_data_folder, "*.png")))

for idx in range(len(gt_filenames)):
    gt_image = Image.open(gt_filenames[idx])
    generated_image = Image.open(generated_filenames[idx])

    print(f'------Image:{idx}--------')
    
    # Calculate the MSE by taking the mean of the squared differences
    gt_array = np.array(gt_image)
    generated_array = np.array(generated_image)
    squared_diff = (gt_array - generated_array) ** 2
    mse = np.mean(squared_diff)
    print(f'squared_diff with np.mean : {mse.item()}')

    # Calculate the MSE using nn.MSELoss
    gt_tensor = torch.tensor(gt_array, dtype=torch.float32)  # Convert NumPy array to a PyTorch tensor
    generated_tensor = torch.tensor(generated_array, dtype=torch.float32)
    mse_loss = nn.MSELoss()
    mse = mse_loss(generated_tensor, gt_tensor)
    print(f'nn.MSELoss: {mse.item()}')
