import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from tokenizer import BPETokenizer
from decoder import Decoder, Config, PrefixEncoder
from torch.utils.data import DataLoader, Dataset
import json
import os
import torch.nn.functional as F
import sys
from tqdm import tqdm

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

val_image_folder = sys.argv[1]
output_json_path = sys.argv[2]
decoder_checkpoint = sys.argv[3]

PEFT_train_type = 'adapter'
checkpoint_name = 'adapter_none_4.pth'
encoder_json_path = './encoder.json'
vocab_file_path = './vocab.bpe'

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
])

# Create an instance of the BPETokenizer
tokenizer = BPETokenizer(encoder_file=encoder_json_path, vocab_file=vocab_file_path)

class EncoderDecoder(nn.Module):
    def __init__(self, decoder_checkpoint=None, peft_type="adapter"):
        super(EncoderDecoder, self).__init__()
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
        elif self.peft_type == "prefixTuning":
            cross_n_embd = 384
        self.decoder_cfg = Config(checkpoint=decoder_checkpoint, peft_type=peft_type, cross_n_embd=cross_n_embd)
        if self.peft_type == "prefixTuning":
            self.prefix_encoder = PrefixEncoder(cfg=self.decoder_cfg)
        self.decoder = Decoder(cfg=self.decoder_cfg)
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, images, captions):
        # Forward propagation
        features = self.encoder.forward_features(images)
        past_key_values_prompt = None
        if self.peft_type == "prefixTuning":
            past_key_values_prompt = self.prefix_encoder(batch_size=captions.size(0))
        outputs = self.decoder(captions, features, past_key_values_prompt)
        return outputs

model = EncoderDecoder(decoder_checkpoint=decoder_checkpoint, peft_type=PEFT_train_type)
val_dataset = CustomDataset(val_image_folder, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint_path = os.path.join('./', checkpoint_name)
state_dict = torch.load(checkpoint_path)
print(sum([p.numel() for n, p in state_dict.items()]))

model.load_state_dict(state_dict, strict=False)

filenames = sorted(os.listdir(val_image_folder))
output = {}

model.eval()
for filename in tqdm(filenames):
    with torch.no_grad():
        image_path = os.path.join(val_image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        image = val_transform(image)
        image = image.unsqueeze(0)
        
        start_token = tokenizer.encode('<|endoftext|>', allowed_special=['<|endoftext|>'])[0]
        caption_input = torch.tensor([[start_token]] * 64).transpose(1, 0) .to(device)
        max_len = 60
        image, caption_input = image.to(device), caption_input.to(device)
        features = model.encoder.forward_features(image)
        
        for i in range(max_len-1):
            past_key_values_prompt = None
            if PEFT_train_type == "prefixTuning":
                past_key_values_prompt = model.prefix_encoder(batch_size=caption_input.size(0))
            predictions = model.decoder(caption_input, features, past_key_values_prompt)
            #predictions = model.decoder(caption_input, features)
            predictions = predictions[:, i, :]
            next_char = torch.argmax(predictions, axis=-1)
            if next_char[0] == 50256:
                break
            caption_input[:, i+1] = next_char
        predicted_captions = [tokenizer.decode(tokens.tolist()[1:]) for tokens in caption_input[:,:i+1]]
        output[filename.split('.')[0]] = predicted_captions[0]
        
with open(output_json_path, 'w') as f:
    json.dump(output, f)
