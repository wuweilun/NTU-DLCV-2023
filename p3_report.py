import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from tokenizer import BPETokenizer
from decoder import Decoder, Config
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize
import json
import os
import torch.nn.functional as F
import loralib as lora
import sys
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as tud

PEFT_train_type = 'adapter'
encoder_json_path = './encoder.json'
vocab_file_path = './vocab.bpe'
decoder_checkpoint = './hw3_data/p2_data/decoder_model.bin'

class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.images = sorted(os.listdir(image_folder))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Read image file
        image_path = os.path.join(self.image_folder, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        key = os.path.splitext(self.images[idx])[0]
        if self.transform:
            image = self.transform(image)
        return image, key

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
])

# Create an instance of the BPETokenizer
tokenizer = BPETokenizer(encoder_file=encoder_json_path, vocab_file=vocab_file_path)

class EncoderDecoder(nn.Module):
    def __init__(self, decoder_checkpoint=None, peft_type="adapter"):
        super(EncoderDecoder, self).__init__()
        #model_name = 'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k'
        #model_name = 'vit_large_patch14_clip_224.openai_ft_in12k_in1k'
        model_name = 'vit_large_patch14_clip_224.openai'
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        for param in self.encoder.parameters():
            param.requires_grad = False
        for submodule in self.encoder.children():
            for param in submodule.parameters():
                param.requires_grad = False

        self.peft_type = peft_type
        if self.peft_type == "adapter":
            cross_n_embd = 384
        elif self.peft_type == "lora":
            cross_n_embd = 768
        self.decoder_cfg = Config(checkpoint=decoder_checkpoint, peft_type=peft_type, cross_n_embd=cross_n_embd, atten_map_flag=True)
        self.decoder = Decoder(cfg=self.decoder_cfg)
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, images, captions):
        # Forward propagation
        past_key_values_prompt = None
        features = self.encoder.forward_features(images)
        outputs = self.decoder(captions, features, past_key_values_prompt)
        return outputs
    
# Visualization of Attention in five test images
val_image_folder = 'hw3_data/p3_data/images/'
val_dataset = CustomDataset(val_image_folder, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

checkpoint_name = 'adapter_none_4.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoder(decoder_checkpoint=decoder_checkpoint, peft_type=PEFT_train_type).to(device)
model.eval()

checkpoint_path = os.path.join('./', checkpoint_name)
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict, strict=False)

predictions = {}
with torch.no_grad():
    for images, images_id in val_dataloader:
        images, images_id = images.to(device), images_id
        start_token = tokenizer.encode('<|endoftext|>', allowed_special=['<|endoftext|>'])[0]
        caption_input = torch.tensor([[start_token]] * 64).transpose(1, 0) .to(device)
        max_caption_length = 30
        fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(16, 8))
        ax[0][0].imshow(images.squeeze(0).permute(1, 2, 0).cpu().numpy())
        ax[0][0].set_title('<|endoftext|>')
        for i in range(3):
            for j in range(5):
                ax[i][j].axis('off')
        max_len = 30
        features = model.encoder.forward_features(images)
        for i in range(max_len-1):
            predictions, atten_map = model.decoder(caption_input, features)
            predictions = predictions[:, i, :]
            next_char = torch.argmax(predictions, axis=-1)

            atten_map = atten_map.squeeze(1)[-1].cpu()
            #print(atten_map.shape)
            atten_map_head = atten_map[i, 1:]
            #print(atten_map_head.shape)
            atten_map_head = torch.reshape(atten_map_head, (16, 16))
            #atten_map_head =  (atten_map_head - atten_map_head.min()) / (atten_map_head.max() - atten_map_head.min())
            atten_map_head =  (atten_map_head - torch.min(atten_map_head)) / (torch.max(atten_map_head) - torch.min(atten_map_head))
            #print(atten_map_head.shape)
            atten_map_head = resize(atten_map_head.unsqueeze(0), [224,224]).squeeze(0)
            #atten_map_head = F.interpolate(atten_map_head.view(1, 1, 16, 16), (224, 224), mode='bilinear').view(224, 224, 1)
            ax[(i+1) // 5][(i+1) % 5].imshow(np.transpose(images.cpu()[0].numpy(), (1, 2, 0)))
            ax[(i+1) // 5][(i+1) % 5].imshow(atten_map_head, alpha=0.6, cmap='jet')
            ax[(i+1) // 5][(i+1) % 5].set_title(tokenizer.decode(next_char.cpu().tolist()))
            if next_char[0] == 50256:
                break
            caption_input[:, i+1] = next_char
        predicted_captions = [tokenizer.decode(tokens.tolist()[1:]) for tokens in caption_input[:,:i+1]]
        plt.savefig(f'{images_id[0]}.jpg')
        
# Visualization of Attention in Top-1 and last-1 image-caption pairs with CLIPScore
from p2_evaluate import CLIPScore
from tqdm import tqdm
val_image_folder = 'hw3_data/p2_data/images/val'
val_dataset = CustomDataset(val_image_folder, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

all_output = {}
best_clip_score = 0
worst_clip_score = 1
calCLIP = CLIPScore()
with torch.no_grad():
    for images, images_id in tqdm(val_dataloader):
        output = {}
        images = images.to(device)
        start_token = tokenizer.encode('<|endoftext|>', allowed_special=['<|endoftext|>'])[0]
        caption_input = torch.tensor([[start_token]] * 64).transpose(1, 0).to(device)
        max_len = 30
        features = model.encoder.forward_features(images)
        for i in range(max_len-1):
            predictions, atten_map = model.decoder(caption_input, features)
            predictions = predictions[:, i, :]
            next_char = torch.argmax(predictions, axis=-1)

            if next_char[0] == 50256:
                break
            caption_input[:, i+1] = next_char
        predicted_captions = [tokenizer.decode(tokens.tolist()[1:]) for tokens in caption_input[:,:i+1]]
        #print(predicted_captions[0])
        for img_id, pred_caption in zip(images_id, predicted_captions):
            output[str(img_id)] = pred_caption
            all_output[str(img_id)] = pred_caption
        clip_score = calCLIP(output, val_image_folder)
        if clip_score > best_clip_score:
            best_clip_score = clip_score
            best_image_id = str(images_id[0])
        if clip_score < worst_clip_score:
            worst_clip_score = clip_score
            worst_image_id = str(images_id[0])
    print(' ')
    print(f'Best image id: {best_image_id}, Clip score: {best_clip_score}')
    print(f'Worst image id: {worst_image_id}, Clip score: {worst_clip_score}')
    
filenames = [best_image_id, worst_image_id]
image_num=0
with torch.no_grad():
    for filename in filenames:
        image_path = os.path.join(val_image_folder, f'{filename}.jpg')
        image = Image.open(image_path).convert("RGB")
        image = val_transform(image)
        image = image.unsqueeze(0)
        images = image.to(device)
        start_token = tokenizer.encode('<|endoftext|>', allowed_special=['<|endoftext|>'])[0]
        caption_input = torch.tensor([[start_token]] * 64).transpose(1, 0).to(device)

        fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(16, 8))
        ax[0][0].imshow(images.squeeze(0).permute(1, 2, 0).cpu().numpy())
        ax[0][0].set_title('<|endoftext|>')
        for i in range(3):
            for j in range(5):
                ax[i][j].axis('off')
        max_len = 30
        features = model.encoder.forward_features(images)
        for i in range(max_len-1):
            predictions, atten_map = model.decoder(caption_input, features)
            predictions = predictions[:, i, :]
            next_char = torch.argmax(predictions, axis=-1)

            atten_map = atten_map.squeeze(1)[-1].cpu()
            atten_map_head = atten_map[i, 1:]
            atten_map_head = torch.reshape(atten_map_head, (16, 16))
            atten_map_head =  (atten_map_head - atten_map_head.min()) / (atten_map_head.max() - atten_map_head.min())
            atten_map_head = resize(atten_map_head.unsqueeze(0), [224,224]).squeeze(0)

            ax[(i+1) // 5][(i+1) % 5].imshow(np.transpose(images.cpu()[0].numpy(), (1, 2, 0)))
            ax[(i+1) // 5][(i+1) % 5].imshow(atten_map_head, alpha=0.6, cmap='jet')
            ax[(i+1) // 5][(i+1) % 5].set_title(tokenizer.decode(next_char.cpu().tolist()))
            if next_char[0] == 50256:
                break
            caption_input[:, i+1] = next_char
        if image_num==0:
            plt.savefig(f'best.jpg')
        else:
            plt.savefig(f'worst.jpg')
        image_num+=1
        