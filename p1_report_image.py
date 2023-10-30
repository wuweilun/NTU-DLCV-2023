# DDPM architecture: https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py

import os
import numpy as np

import torch
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from tqdm import tqdm

import sys
from unet_conditional import UNet_conditional
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Path to the figure folders
figure_folder = sys.argv[1]

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=28, num_classes=10, c_in=3, c_out=3):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = (1. - self.beta).to(device)
        self.alpha_hat = (torch.cumprod(self.alpha, dim=0)).to(device)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes).to(device)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample(self, labels, cfg_scale=3):
        model = self.model
        n = len(labels)
        model.eval()
        with torch.cuda.amp.autocast():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            
            image = x[0]
            image = transforms.Resize((28, 28), antialias=True)(image)
            grid = make_grid(image, nrow=1)
            filename = f"class_0_timesteps_0.png"
            image_path = os.path.join(figure_folder, filename)
            save_image(grid, image_path) 
            
            timesteps_list = [200, 400, 600, 800, 1000]
            for i in tqdm(reversed(range(1, self.noise_steps))):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
                timestep = 1000-i+1
                if timestep in timesteps_list:
                    image = x[0]
                    image = transforms.Resize((28, 28), antialias=True)(image)
                    grid = make_grid(image, nrow=1)
                    filename = f"class_0_timesteps_{timestep}.png"
                    image_path = os.path.join(figure_folder, filename)
                    save_image(grid, image_path) 
        return x

ddpm = Diffusion(img_size=32)
scaler = torch.cuda.amp.GradScaler()

checkpoint_name = 'ddpm_499.pth'
checkpoint_path = os.path.join('./model_checkpoint/', checkpoint_name)
checkpoint_info = torch.load(checkpoint_path)['model_state_dict']

ddpm.model.load_state_dict(checkpoint_info)

ddpm.model.eval()  # Set the model to evaluation mode

num_classes = 10
num_samples_per_class = 10
labels = []

for sample_index in range(num_samples_per_class):
    for class_label in range(num_classes):
        labels.append(sample_index)

labels = torch.tensor(labels, dtype=torch.int).to(device)

with torch.no_grad():
    sampled_images = ddpm.sample(labels=labels)

    sampled_images = transforms.Resize((28*10, 28*10), antialias=True)(sampled_images)
    grid = make_grid(sampled_images, nrow=10)
    filename = f"grid_each_digits.png"
    image_path = os.path.join(figure_folder, filename)
    save_image(grid, image_path)

