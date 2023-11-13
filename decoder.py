import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.adapter_size = int(self.n_embd * 0.25)

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
    
class CrossAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn_text = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_attn_image = nn.Linear(1024, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x_text, x_image):
        B, T_text, C_text = x_text.size()  # batch, context_text, embedding_text
        _, T_image, C_image = x_image.size()  # batch, context_image, embedding_image
        # print(B, T_text, C_text)
        # print(_, T_image, C_image)
        q_text, k_text, v_text = self.c_attn_text(x_text).split(self.n_embd, dim=2)
        # print(q_text.shape)
        #k_text = k_text.view(B, T_text, self.n_head, C_text // self.n_head).transpose(1, 2)
        q_text = q_text.view(B, T_text, self.n_head, C_text // self.n_head).transpose(1, 2)
        #v_text = v_text.view(B, T_text, self.n_head, C_text // self.n_head).transpose(1, 2)
        # print(q_text.shape)
        
        # Maybe change C_text to C_images? 
        q_image, k_image, v_image = self.c_attn_image(x_image).split(self.n_embd, dim=2)
        # print(k_image.shape, v_image.shape)
        k_image = k_image.view(B, T_image, self.n_head, C_text // self.n_head).transpose(1, 2) 
        # q_image = q_image.view(B, T_image, self.n_head, C_image // self.n_head).transpose(1, 2)
        v_image = v_image.view(B, T_image, self.n_head, C_text // self.n_head).transpose(1, 2)
        # print(k_image.shape, v_image.shape)
        
        att = (q_text @ k_image.transpose(-2, -1)) * (1.0 / math.sqrt(k_image.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T_text, :T_image] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        return self.c_proj((att @ v_image).transpose(1, 2).contiguous().view(B, T_text, C_text))
    
class AdapterLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(AdapterLayer, self).__init__()
        self.downsample = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.upsample = nn.Linear(output_size, input_size)

    def forward(self, x):
        downsampled = self.downsample(x)
        activated = self.relu(downsampled)
        upsampled = self.upsample(activated)
        return x + upsampled
    
class BlockAdapter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.adapter_layer_1 = AdapterLayer(cfg.n_embd, cfg.adapter_size)
        self.adapter_layer_2 = AdapterLayer(cfg.n_embd, cfg.adapter_size)
        self.adapter_layer_3 = AdapterLayer(cfg.n_embd, cfg.adapter_size)
        self.attn = Attention(cfg)
        self.attn_cross = CrossAttention(cfg)  # cross-attention between text and image
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, x, x_image):
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.attn_cross(self.ln_2(x), x_image)
        # x = x + self.mlp(self.ln_3(x))
        
        x = x + self.adapter_layer_1(self.attn(self.ln_1(x)))
        x = x + self.adapter_layer_2(self.attn_cross(self.ln_2(x), x_image))
        x = x + self.adapter_layer_3(self.mlp(self.ln_3(x)))
        return x
    
class DecoderAdapter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            # h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            # h = n_layer(self.cfg),
            h = nn.ModuleList([BlockAdapter(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x, x_image):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        # x = self.lm_head(self.transformer.ln_f(self.transformer.h(x, x_image)))
        for block in self.transformer.h:
            x = block(x, x_image)
        # x = self.transformer.h(x, x_image)
        x = self.lm_head(self.transformer.ln_f(x))
        return x
