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

seed = 54
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        self.interpolation_noise = None
        
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
    
    def sample(self, time_steps=50, eta=0, stop_idx=4, interpolation=False):
        model = self.model
        model.eval()
        if interpolation:
            noises = [self.interpolation_noise]
        else:
            noises = sorted(glob.glob(os.path.join(noises_folder, "*.pt")))
        noise_idx = 0
        interval = self.noise_steps // time_steps
        seq = range(0, self.noise_steps, interval)
        seq = [int(s) for s in list(seq)]
        
        with torch.no_grad():
            for noise in noises:
                if interpolation:
                    x = noise
                else:
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
                # min_value = torch.min(image)
                # max_value = torch.max(image)

                # # Perform min-max normalization
                # image_normalized = (image - min_value) / (max_value - min_value)
                # image = (image_normalized * 255).type(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
                # #print(image.shape)
                # filename = f"{noise_idx:02d}_{eta}.png"
                # image_path = os.path.join(figure_folder, filename)
                # self.image_path.append(image_path)
                
                # # Convert the NumPy array to a PIL image
                # image = Image.fromarray(np.uint8(image))
                # image.save(image_path)
                # noise_idx+=1
                if interpolation:
                    return image
                else:
                    image = image.clamp(-1, 1)
                    filename = f"{noise_idx:02d}_eta_{eta}.png"
                    image_path = os.path.join(figure_folder, filename)
                    self.image_path.append(image_path)
                    save_image(image, image_path, normalize=True)
                    noise_idx+=1
                
                if stop_idx == noise_idx:
                    break
                
    def interpolation(self, method=None, noise0_name="00.pt", noise1_name="01.pt", alpha=None):
        noise1_path = os.path.join(noises_folder, noise0_name)
        noise2_path = os.path.join(noises_folder, noise1_name)
        x0 = torch.load(noise1_path)
        x1 = torch.load(noise2_path)
        
        if method=='spherical':
            theta = torch.acos(torch.sum(x0 * x1)/(torch.norm(x0) * torch.norm(x1)))
            interpolated_noise = torch.sin((1 - alpha) * theta) / torch.sin(theta) * x0 + torch.sin(alpha * theta) / torch.sin(theta) * x1
            self.interpolation_noise = interpolated_noise
            
            filename = f"spherical_linear_interpolation_{noise0_name.replace('.pt', '')}_{noise1_name.replace('.pt', '')}_{alpha}.png"
        else:
            interpolated_noise = (1 - alpha) * x0 + alpha * x1
            self.interpolation_noise = interpolated_noise
            
            filename = f"linear_interpolation_{noise0_name.replace('.pt', '')}_{noise1_name.replace('.pt', '')}_{alpha}.png"
        
        image = self.sample(time_steps=50, eta=0, interpolation=True)
        image = image.clamp(-1, 1)
        image_path = os.path.join(figure_folder, filename)
        self.image_path.append(image_path)
        save_image(image, image_path, normalize=True)

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
        ddpm.interpolation(method='spherical', noise0_name="00.pt", noise1_name="01.pt", alpha=alpha)
    ddpm.concat_images(result_filename='slerp.png', row=1, col=11)  
    ddpm.image_path = [] 
    
    for alpha in alpha_list:
        ddpm.interpolation(method='linear', noise0_name="00.pt", noise1_name="01.pt", alpha=alpha)
    ddpm.concat_images(result_filename='linear_interpolation.png', row=1, col=11)
    ddpm.image_path = []   
    

