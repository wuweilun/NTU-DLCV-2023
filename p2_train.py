import torch
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
import math
from PIL import Image
from torch import nn
import torch.optim as optim
from tokenizer import BPETokenizer
from decoder import Decoder, Config
from tqdm import tqdm
import json
import os
from p2_evaluate import CIDERScore, getGTCaptions
import torch.nn.functional as F
import loralib as lora
from torch.cuda.amp import autocast, GradScaler
import sys
from scheduler import CosineAnnealingWarmupRestarts
import warnings 
warnings.filterwarnings('ignore') 

class CustomDataset(Dataset):
    def __init__(self, annotations, images, image_folder, tokenizer, transform=None):
        self.annotations = annotations
        self.images = {img['id']: img['file_name'] for img in images}
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # Read image file
        image_id = annotation.get('image_id', '')
        image_path = os.path.join(self.image_folder, self.images[image_id])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        caption = annotation.get('caption', '')
        caption_tokens = self.tokenizer.encode(caption, allowed_special=['<|endoftext|>'])
        caption_tokens = [50256] + caption_tokens
        
        # Pad captions to the same length
        max_caption_length = 64
        if len(caption_tokens) < max_caption_length:
            caption_tokens += [50256] * (max_caption_length - len(caption_tokens))
        else:
            caption_tokens = caption_tokens[:max_caption_length]

        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)
        
        return image, caption_tokens, image_id
    
PEFT_train_type = sys.argv[1]
train_json_path = './hw3_data/p2_data/train.json'
train_image_folder = 'hw3_data/p2_data/images/train'
val_json_path = './hw3_data/p2_data/val.json'
val_image_folder = 'hw3_data/p2_data/images/val'
encoder_json_path = './encoder.json'
vocab_file_path = './vocab.bpe'
decoder_checkpoint = './hw3_data/p2_data/decoder_model.bin'

# Create an instance of the BPETokenizer
tokenizer = BPETokenizer(encoder_file=encoder_json_path, vocab_file=vocab_file_path)

# Load json
with open(train_json_path, 'r') as f:
    train_data = json.load(f)
    
with open(val_json_path, 'r') as f:
    val_data = json.load(f)

val_gts = getGTCaptions(val_data)
train_transform = transforms.Compose([
    #transforms.Resize((336,336)),  # Adjust size as needed 336
    transforms.Resize((224,224)),  
    transforms.AutoAugment(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    #transforms.Resize((336,336)),  # Adjust size as needed
    transforms.Resize((224,224)),  
    transforms.ToTensor(),
])

train_dataset = CustomDataset(annotations=train_data['annotations'], images=train_data['images'], image_folder= train_image_folder, tokenizer=tokenizer, transform=train_transform)
sampler_train_dataset = torch.utils.data.RandomSampler(train_dataset)
batch_sampler_train_dataset = torch.utils.data.BatchSampler(sampler_train_dataset, batch_size=32, drop_last=True)
train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train_dataset, num_workers=4)

val_dataset = CustomDataset(annotations=val_data['annotations'], images=val_data['images'], image_folder= val_image_folder, tokenizer=tokenizer, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# tmp=0
# for annotation in train_data['annotations']:
#     caption = annotation.get('caption', '')
#     caption_tokens = tokenizer.encode(caption, allowed_special=[''])
#     #print(len(caption_tokens))
#     if  64 < len(caption_tokens):
#         tmp+=1
# print(tmp)   
class EncoderDecoder(nn.Module):
    def __init__(self, decoder_checkpoint=None, peft_type="adapter"):
        super(EncoderDecoder, self).__init__()
        #model_name = 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k'
        #model_name = 'vit_large_patch14_clip_224.openai_ft_in12k_in1k'
        #model_name = 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k' #no use
        #model_name = 'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k' # base line:CIDEr: 0.7633182475486938 | CLIPScore: 0.6982138704627235
        #model_name = 'vit_large_patch14_clip_336.openai_ft_in12k_in1k'
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
        self.decoder_cfg = Config(checkpoint=decoder_checkpoint, peft_type=peft_type, cross_n_embd=cross_n_embd)
        self.decoder = Decoder(cfg=self.decoder_cfg)
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, images, captions):
        # Forward propagation
        features = self.encoder.forward_features(images)
        outputs = self.decoder(captions, features)
        return outputs

model = EncoderDecoder(decoder_checkpoint=decoder_checkpoint, peft_type=PEFT_train_type)
if model.peft_type == "lora":
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
for block in model.decoder.transformer.h:
    if model.peft_type == "adapter":
        for param in block.adapter_layer_1.parameters():
            param.requires_grad = True
        # for param in block.adapter_layer_2.parameters():
        #     param.requires_grad = True
        for param in block.adapter_layer_3.parameters():
            param.requires_grad = True
        for param in block.ln_1.parameters():
            param.requires_grad = True
        for param in block.ln_2.parameters():
            param.requires_grad = True
        for param in block.ln_3.parameters():
            param.requires_grad = True
            
    for param in block.attn_cross.parameters():
        param.requires_grad = True
        
# Place Encoder and Decoder on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation setup
best_val_loss = 100.0

epoch = 0
epochs = 20
checkpoint = False
checkpoint_name = 'Adapter_best_9.pth'
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)
scaler = GradScaler()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-100)
#optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # lr=1e-4
# optimizer = optim.AdamW(model.parameters(), lr=1e-4)
#optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-9)
total_steps = len(train_dataloader) * epochs
#print(total_steps)
warmup_steps = 0.1 * total_steps  
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=total_steps,
    cycle_mult=1.0,
    max_lr=5e-4,
    min_lr=1e-6,
    warmup_steps=warmup_steps,
    gamma=1.0,
)

# max_cosine_lr_factor = 0.9 
# # Learning Rate scheduing with warmup and cosine annealing
# def lr_lambda(current_step):
#     if current_step < warmup_steps:
#         return float(current_step) / float(max(1, warmup_steps))
#     else:
#         return max(
#             0.0, max_cosine_lr_factor * 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
#         )

# scheduler = LambdaLR(optimizer, lr_lambda)

if checkpoint is True:
    # # load checkpoint
    # checkpoint_info = torch.load(checkpoint_path)
    # epoch = checkpoint_info['epoch'] + 1
    # model.load_state_dict(checkpoint_info['model_state_dict'])
    state_dict = torch.load(checkpoint_path)
    # model.load_state_dict(state_dict['layer_state_dicts'], strict=False)
    # model.load_state_dict(state_dict['peft_state_dicts'], strict=False)
    model.load_state_dict(state_dict, strict=False)
    # optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
    
print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# Training loop
while epoch < epochs:
    model.train()  # Set the model to training mode
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    
    for images, caption_tokens, _ in progress_bar:
        images, captions = images.to(device), caption_tokens.to(device)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images, captions)

            # Flatten caption_length and batch_size to get logits with shape [batch_size * caption_length, vocab_size]
            outputs = outputs[:, :-1,:].contiguous()
            #print(outputs.shape)
            logits_flat = outputs.reshape(-1, 50257)
            caption_length=63
            
            # Flatten caption_length and batch_size to get targets with shape [batch_size * caption_length]
            captions = captions[:,1:].contiguous()
            for i in range(captions.size(0)):
                end_token_indices = (captions[i] == 50256).nonzero()
                if end_token_indices.numel() > 0:
                    col = end_token_indices[0]
                    if col < captions.size(1) - 1:
                        captions[i, col + 1:] = -100
            targets_flat = captions.reshape(-1)

            loss = criterion(logits_flat, targets_flat)
            if torch.isnan(loss).any():
                print("NaN detected in loss!")
            #print(loss)
        scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()
        
    # Calculate average training loss
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Training Loss: {avg_train_loss:.4f}")
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, caption_tokens, _ in tqdm(val_dataloader, desc="Validation"):
            images, captions = images.to(device), caption_tokens.to(device)
            with autocast():
                outputs = model(images, captions)
                # Flatten caption_length and batch_size to get logits with shape [batch_size * caption_length, vocab_size]
                outputs = outputs[:, :-1,:].contiguous()
                #print(outputs.shape)
                logits_flat = outputs.reshape(-1, 50257)
                caption_length=63
                
                # Flatten caption_length and batch_size to get targets with shape [batch_size * caption_length]
                captions = captions[:,1:].contiguous()
                for i in range(captions.size(0)):
                    end_token_indices = (captions[i] == 50256).nonzero()
                    if end_token_indices.numel() > 0:
                        col = end_token_indices[0]
                        if col < captions.size(1) - 1:
                            captions[i, col + 1:] = -100
                targets_flat = captions.reshape(-1)

                loss = criterion(logits_flat, targets_flat)
            val_loss += loss.item()
        # Calculate average training loss
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    # layer_state_dicts = {}
    # layer_list = ['.query', '.key', '.value', 'c_proj_cross']
    # for name, weight in model.state_dict().items():
    #     if any(w in name for w in layer_list):
    #         #print(name)
    #         layer_state_dicts[name] = weight
    # peft_state_dicts = {}
    # if model.peft_type == "lora":
    #     peft_state_dicts = lora.lora_state_dict(model, bias='lora_only')
    # elif model.peft_type == "adapter":
    #     peft_list = [ '.downsample', '.upsample', '.ln_1', '.ln_2', '.ln_3']
    #     for name, weight in model.state_dict().items():
    #         if any(w in name for w in peft_list):
    #             peft_state_dicts[name] = weight
    trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
    save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
    # Save the model with the best CIDEr score
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # checkpoint = {
        #     'layer_state_dicts': layer_state_dicts,
        #     'peft_state_dicts': peft_state_dicts,
        # }
        checkpoint_path = os.path.join('./model_checkpoint', f'{PEFT_train_type}_best_{epoch}.pth')
        torch.save(save_weights, checkpoint_path)
    else:
        # checkpoint = {
        #     'epoch': epoch,
        #     'layer_state_dicts': layer_state_dicts,
        #     'peft_state_dicts': peft_state_dicts,
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        # }
        checkpoint_path = os.path.join('./model_checkpoint', f'{PEFT_train_type}_{epoch}.pth')
        torch.save(save_weights, checkpoint_path)
    epoch+=1