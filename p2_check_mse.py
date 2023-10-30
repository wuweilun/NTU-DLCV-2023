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

mean_np_mse = []
mean_torch_mse = []
for idx in range(len(gt_filenames)):
    gt_image = Image.open(gt_filenames[idx])
    generated_image = Image.open(generated_filenames[idx])

    print(f'------Image:{idx}--------')
    
    # Calculate the MSE by taking the mean of the squared differences
    gt_array = np.array(gt_image)
    generated_array = np.array(generated_image)
    squared_diff = (gt_array - generated_array) ** 2
    mse = np.mean(squared_diff)
    mean_np_mse.append(mse)
    print(f'squared_diff with np.mean : {mse.item()}')

    # Calculate the MSE using nn.MSELoss
    gt_tensor = torch.tensor(gt_array, dtype=torch.float32)  # Convert NumPy array to a PyTorch tensor
    generated_tensor = torch.tensor(generated_array, dtype=torch.float32)
    mse_loss = nn.MSELoss()
    mse = mse_loss(generated_tensor, gt_tensor)
    mean_torch_mse.append(mse.item())
    print(f'nn.MSELoss: {mse.item()}')

print(f'10 images with mean(squared_diff with np.mean): {np.mean(mean_np_mse)}')
print(f'10 images with mean(nn.MSELoss): {np.mean(mean_torch_mse)}')
