# DDPM architecture: https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py
# sampling formulas: https://github.com/ermongroup/ddim/blob/main/functions/denoising.py#L10

import os
import numpy as np

import torch
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from tqdm import tqdm

import sys
from UNet import UNet
from utils import beta_scheduler
import random
import glob 
from PIL import Image

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Path to the predefined noises, figure folders, model weight
noises_folder = sys.argv[1]
figure_folder = sys.argv[2]
checkpoint_path = sys.argv[3]

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

class Diffusion:
    def __init__(self, noise_steps=1000,img_size=28, c_in=3):
        self.noise_steps = noise_steps

        self.beta = beta_scheduler().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet().to(device)
        self.device = device
        self.c_in = c_in
        self.image_path = []
        
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    def sample(self, time_steps=50, eta=0, stop_idx=4):
        model = self.model
        model.eval()
        noises = sorted(glob.glob(os.path.join(noises_folder, "*.pt")))
        noise_idx = 0
        interval = self.noise_steps // time_steps
        seq = range(0, self.noise_steps, interval)
        seq = [int(s) for s in list(seq)]
        
        with torch.no_grad():
            for noise in noises:
                x = torch.load(noise).to(self.device)
                n = x.size(0)
                seq_next = [-1] + list(seq[:-1])
                x0_preds = []
                xs = [x]
                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = self.compute_alpha(self.beta, t.long())
                    at_next = self.compute_alpha(self.beta, next_t.long())
                    xt = xs[-1].to('cuda')
                    et = model(xt, t)
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    x0_preds.append(x0_t.to('cpu'))
                    c1 = (
                        eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    )
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                    xt_next = xt_next.float()
                    xs.append(xt_next.to('cpu'))
                    
                x = xs[-1]
                image = x[0]
                min_value = torch.min(image)
                max_value = torch.max(image)

                # Perform min-max normalization
                image_normalized = (image - min_value) / (max_value - min_value)
                image = (image_normalized * 255).type(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
                #print(image.shape)
                filename = f"{noise_idx:02d}_{eta}.png"
                image_path = os.path.join(figure_folder, filename)
                self.image_path.append(image_path)
                
                # Convert the NumPy array to a PIL image
                image = Image.fromarray(np.uint8(image))
                image.save(image_path)
                noise_idx+=1
                if stop_idx == noise_idx:
                    break
                
    def interpolation(self, method=None, image_folder=None, image1_name=None, image2_name=None, alpha=None):
        image1_path = os.path.join(image_folder, image1_name)
        image2_path = os.path.join(image_folder, image2_name)
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        
        if method=='spherical':
            # transform = transforms.ToTensor()
            # x0 = (transform(image1)* 255)
            # x1 = (transform(image2)* 255)
            # theta = torch.acos(torch.sum(x0 * x1)/(torch.norm(x0) * torch.norm(x1)))
            # image = torch.sin((1 - alpha) * theta) / torch.sin(theta) * x0 + torch.sin(alpha * theta) / torch.sin(theta) * x1
            # image = image.type(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
            
            v0 = image1
            v1 = image2
            # Copy the vectors to reuse them later
            v0_copy = np.copy(v0)
            v1_copy = np.copy(v1)
            # Normalize the vectors to get the directions and angles
            v0 = v0 / np.linalg.norm(v0)
            v1 = v1 / np.linalg.norm(v1)
            # Dot product with the normalized vectors (can't use np.dot in W)
            dot = np.sum(v0 * v1)
            theta = np.arccos(dot)

            # Finish the slerp algorithm
            s0 = np.sin((1-alpha) * theta) / np.sin(theta)
            s1 = np.sin(alpha * theta) / np.sin(theta)
            image = s0 * v0_copy + s1 * v1_copy    
            filename = f"spherical_linear_interpolation_{image1_name.replace('.png', '')}_{image2_name.replace('.png', '')}_{alpha}.png"
        else:
            array1 = np.array(image1)
            array2 = np.array(image2)
            image = (1 - alpha) * array1 + alpha * array2
            filename = f"linear_interpolation_{image1_name.replace('.png', '')}_{image2_name.replace('.png', '')}_{alpha}.png"
            
        image_path = os.path.join(figure_folder, filename)
        self.image_path.append(image_path)
        
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(np.uint8(image))
        image.save(image_path)

    def concat_images(self, result_filename=None, row=None, col=None):
        images = [Image.open(filename) for filename in self.image_path]
        width, height = images[0].size

        total_width = col * width
        total_height = row * height

        result_image = Image.new('RGB', (total_width, total_height))

        for i, img in enumerate(images):
            x = (i % col ) * width
            y = (i // col) * height
            result_image.paste(img, (x, y))
            
        image_path = os.path.join(figure_folder, result_filename)
        result_image.save(image_path)

ddpm = Diffusion(img_size=32)

checkpoint_info = torch.load(checkpoint_path)
ddpm.model.load_state_dict(checkpoint_info)

ddpm.model.eval()  # Set the model to evaluation mode

eta_list =  [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_list = [i / 10.0 for i in range(11)]

with torch.no_grad():
    for eta in eta_list:
        ddpm.sample(time_steps=50, eta=eta, stop_idx=4)
    ddpm.concat_images(result_filename='different_eta.png', row=5, col=4)
    ddpm.image_path = []
    
    for alpha in alpha_list:
        ddpm.interpolation(method='spherical', image_folder=figure_folder, image1_name="00.png", image2_name="01.png", alpha=alpha)
    ddpm.concat_images(result_filename='slerp.png', row=1, col=10)  
    ddpm.image_path = [] 
    
    for alpha in alpha_list:
        ddpm.interpolation(method='linear', image_folder=figure_folder, image1_name="00.png", image2_name="01.png", alpha=alpha)
    ddpm.concat_images(result_filename='linear_interpolation.png', row=1, col=10)
    ddpm.image_path = []   
    

