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
    
    def sample(self, time_steps=50, eta=0):
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
                filename = f"{noise_idx:02d}.png"
                image_path = os.path.join(figure_folder, filename)
                
                # Convert the NumPy array to a PIL image
                image = Image.fromarray(np.uint8(image))
                image.save(image_path)
                noise_idx+=1
                
        #         for i in tqdm(reversed(range(1, self.noise_steps))):
        #             t = (torch.ones(1) * i).long().to(self.device)
        #             predicted_noise = model(x, t)
        #             alpha = self.alpha[t][:, None, None, None]
        #             one_step_t = (torch.ones(1) * (i-1)).long().to(self.device)
        #             alpha_one_step = self.alpha[one_step_t][:, None, None, None]
        #             dev = eta * torch.sqrt((1-alpha_one_step)/(1-alpha)) * torch.sqrt((1-alpha/alpha_one_step))
        #             #alpha_hat = self.alpha_hat[t][:, None, None, None]
        #             #beta = self.beta[t][:, None, None, None]
        #             if i > 1:
        #                 noise = torch.randn_like(x)
        #             else:
        #                 noise = torch.zeros_like(x)
        #             x = (torch.sqrt(alpha_one_step) * (x-torch.sqrt(1-alpha)*predicted_noise) / torch.sqrt(alpha)) + (torch.sqrt(1-alpha_one_step-dev*dev) * predicted_noise) +  dev * noise
        #             x = x.float()

ddpm = Diffusion(img_size=32)
#scaler = torch.cuda.amp.GradScaler()

checkpoint_info = torch.load(checkpoint_path)
ddpm.model.load_state_dict(checkpoint_info)

ddpm.model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    ddpm.sample(time_steps=50, eta=0)
