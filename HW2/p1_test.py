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

seed = 54
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
            
        return x

ddpm = Diffusion(img_size=32)
scaler = torch.cuda.amp.GradScaler()

checkpoint_name = 'ddpm_999.pth'
checkpoint_path = os.path.join('./', checkpoint_name)
checkpoint_info = torch.load(checkpoint_path)['model_state_dict']

ddpm.model.load_state_dict(checkpoint_info)

ddpm.model.eval()  # Set the model to evaluation mode

class_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

class_pairs = [(0, 1, False), (2, 3, False), (6, 5, False),
                (4, 7, False), (8, 9, False)]

with torch.no_grad():
    for class1, class2, difficult in class_pairs:
        labels = torch.cat([torch.tensor([class1, class2]).long()] * 100, dim=0).to(device)
        if difficult == True:
            sampled_images = ddpm.sample(labels=labels, cfg_scale=3)
        else:
            sampled_images = ddpm.sample(labels=labels, cfg_scale=0)
        for class_label in (class1, class2):
            for image_number in range(100):  
                image = sampled_images[image_number*2+class_label%2]
                image = transforms.Resize((28, 28), antialias=True)(image)
                grid = make_grid(image, nrow=1)
                filename = f"{class_label}_{image_number+1:03d}.png"
                image_path = os.path.join(figure_folder, filename)
                save_image(grid, image_path)
