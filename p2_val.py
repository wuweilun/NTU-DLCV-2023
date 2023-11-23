import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from tokenizer import BPETokenizer
from decoder import Decoder, Config
from torch.utils.data import DataLoader, Dataset
import json
import os
import torch.nn.functional as F
import loralib as lora
from autoregressive import perform_validation, greedy_search, beam_search
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

PEFT_train_type = sys.argv[1]
checkpoint_name = sys.argv[2]
val_image_folder = 'hw3_data/p2_data/images/val'
encoder_json_path = './encoder.json'
vocab_file_path = './vocab.bpe'
decoder_checkpoint = './hw3_data/p2_data/decoder_model.bin'

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
])

# Create an instance of the BPETokenizer
tokenizer = BPETokenizer(encoder_file=encoder_json_path, vocab_file=vocab_file_path)

# caption = "a kitchen with a sink."
# tokenized_caption = tokenizer.encode(caption, allowed_special=[])
# print(tokenized_caption)
# print(tokenizer.decode(tokenized_caption))
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
        self.decoder_cfg = Config(checkpoint=decoder_checkpoint, peft_type=peft_type, cross_n_embd=cross_n_embd)
        self.decoder = Decoder(cfg=self.decoder_cfg, )
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, images, captions):
        # Forward propagation
        features = self.encoder.forward_features(images)
        outputs = self.decoder(captions, features)
        return outputs
    
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, length, context, start_token, batch_size, temperature, top_k, device, top_p, stop_token, image):
    current_token = start_token
    output = []

    for _ in range(length):
        # Forward pass to get logits
        logits = model(image, current_token)
        print(logits.size())
        logits = logits[:, :min(logits.size(1), 1024)]  # Assuming 40 is the block_size, you can adjust it accordingly
        logits = logits.squeeze()
        logits = logits.squeeze(dim=1)
        print(logits.size())
        filtered_logits = top_k_top_p_filtering(logits, top_k=0, top_p=0.9)

        # Apply temperature
        logits /= temperature

        # Sample from the distribution
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1).squeeze()

        # Append the sampled token to the output
        output.append(next_token.item())

        # Check if the stop token is generated
        if next_token == stop_token:
            break

        # Update the current token for the next iteration
        current_token = next_token

    return output

model = EncoderDecoder(decoder_checkpoint=decoder_checkpoint, peft_type=PEFT_train_type)
# if model.peft_type == "lora":
#     lora.mark_only_lora_as_trainable(model, bias='lora_only')
# for block in model.decoder.transformer.h:
#     if model.peft_type == "adapter":
#         for param in block.adapter_layer_1.parameters():
#             param.requires_grad = True
#         # for param in block.adapter_layer_2.parameters():
#         #     param.requires_grad = True
#         for param in block.adapter_layer_3.parameters():
#             param.requires_grad = True
#     for param in block.attn_cross.parameters():
#         param.requires_grad = True
        
# print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

val_dataset = CustomDataset(val_image_folder, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint_path = os.path.join('./model_checkpoint', checkpoint_name)
# checkpoint_info = torch.load(checkpoint_path)
state_dict = torch.load(checkpoint_path)
print(sum([p.numel() for n, p in state_dict.items()]))

# layer_name = 'decoder.transformer.h.2.ln_1'

# for name, param in model.named_parameters():
#     # print(name)
#     if name == layer_name + '.weight':
#         print(f"Layer: {name}")
#         print(param.data)
#         break
# transposed = [ '.downsample.weight', '.upsample.weight',  
#             '.query.weight', '.key.weight', '.value.weight', 'c_proj_cross.weight']
# for key, value in state_dict['layer_state_dicts'].items():
#     if any(key.endswith(w) for w in transposed):
#         state_dict['layer_state_dicts'][key] = value
# model.load_state_dict(state_dict['layer_state_dicts'], strict=False)
# model.load_state_dict(state_dict['peft_state_dicts'], strict=False)

model.load_state_dict(state_dict, strict=False)
# for name, param in model.named_parameters():
#     #print(name)
#     if name == layer_name + '.weight':
#         print(f"Layer: {name}")
#         print(param.data)
#         break

filenames = sorted(os.listdir(val_image_folder))
output = {}

# val_predictions = perform_validation(model, val_dataloader, device, tokenizer)
# val_predictions = greedy_search(model, val_dataloader, device, tokenizer)
# output = beam_search(model, val_dataloader, device, tokenizer)
# val_predictions = greedy_search(model, val_dataloader, device, tokenizer, 20)

model.eval()
count=0
for filename in tqdm(filenames):
    with torch.no_grad():
        image_path = os.path.join(val_image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        image = val_transform(image)
        image = image.unsqueeze(0)
        
        start_token = tokenizer.encode('<|endoftext|>', allowed_special=['<|endoftext|>'])[0]
        caption_input = torch.tensor([[start_token]] * 64).transpose(1, 0) .to(device)
        #print(caption_input.shape)
        max_len = 30
        image, caption_input = image.to(device), caption_input.to(device)
        features = model.encoder.forward_features(image)
        for i in range(max_len-1):
            predictions = model.decoder(caption_input, features)
            #print(predictions.shape)
            predictions = predictions[:, i, :]
            #print(predictions.shape)
            next_char = torch.argmax(predictions, axis=-1)
            if next_char[0] == 50256:
                break
            caption_input[:, i+1] = next_char
        if i == 28:
            count+=1
        predicted_captions = [tokenizer.decode(tokens.tolist()[1:]) for tokens in caption_input[:,:i+1]]
        #print(predicted_captions[0])
        output[filename.split('.')[0]] = predicted_captions[0]
        #print(filename.split('.')[0])
print(count)
output_json_path = './bash_output/pred.json'
with open(output_json_path, 'w') as f:
    json.dump(output, f)
