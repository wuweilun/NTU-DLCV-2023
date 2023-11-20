import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora

class Config:

    def __init__(self, checkpoint=None, peft_type ="adapter", atten_map_flag=False):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.adapter_size = int(self.n_embd * 0.25)
        self.peft_type = peft_type
        self.dropout = 0.1
        self.atten_map_flag = atten_map_flag
        self.cross_n_embd = 384

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        if cfg.peft_type == "lora":
            self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=4)
        else:
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
    
# class CrossAttention(nn.Module):

#     def __init__(self, cfg):
#         super().__init__()
#         self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
#         #self.c_attn_image = nn.Linear(786, 3 * cfg.n_embd)
#         self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
#         self.n_head = cfg.n_head
#         self.n_embd = cfg.n_embd
#         size = cfg.block_size
#         self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

#     def forward(self, x_text, x_image):
#         B, T_text, C_text = x_text.size()  # batch, context_text, embedding_text
#         _, T_image, C_image = x_image.size()  # batch, context_image, embedding_image
#         #print(B, T_text, C_text)
#         #print(_, T_image, C_image)
        
#         q_text, k_text, v_text = self.c_attn(x_text).split(self.n_embd, dim=2)
#         # print(q_text.shape)
#         #k_text = k_text.view(B, T_text, self.n_head, C_text // self.n_head).transpose(1, 2)
#         q_text = q_text.view(B, T_text, self.n_head, C_text // self.n_head).transpose(1, 2)
#         #v_text = v_text.view(B, T_text, self.n_head, C_text // self.n_head).transpose(1, 2)
#         # print(q_text.shape)
        
#         # Maybe change C_text to C_images? 
#         q_image, k_image, v_image = self.c_attn(x_image).split(self.n_embd, dim=2)
#         # print(k_image.shape, v_image.shape)
#         k_image = k_image.view(B, T_image, self.n_head, C_image // self.n_head).transpose(1, 2) 
#         v_image = v_image.view(B, T_image, self.n_head, C_image // self.n_head).transpose(1, 2)
#         # print(k_image.shape, v_image.shape)
        
#         att = (q_text @ k_image.transpose(-2, -1)) * (1.0 / math.sqrt(k_image.size(-1)))
#         #att = att.masked_fill(self.bias[:, :, :T_text, :T_image] == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)

#         return self.c_proj((att @ v_image).transpose(1, 2).contiguous().view(B, T_text, C_text))

class CrossAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Linear(cfg.n_embd, cfg.cross_n_embd) # how about 384, 192
        self.key = nn.Linear(1024, cfg.cross_n_embd)
        self.value = nn.Linear(1024, cfg.cross_n_embd)
        self.c_proj_cross = nn.Linear(cfg.cross_n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.cross_n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x_text, x_image):
        B, T_text, C_text = x_text.size()  # batch, context_text, embedding_text
        _, T_image, C_image = x_image.size()  # batch, context_image, embedding_image
        #print(B, T_text, T_image) #16 64 197
        #print(B, C_text, C_image)
        
        q_text = self.query(x_text)
        #print(q_text.shape)
        q_text = q_text.view(B, T_text, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        #print(q_text.shape) # ([16, 12, 64, 64])
        
        # Maybe change C_text to C_images? 
        k_image = self.key(x_image)
        #print(k_image.shape)
        v_image = self.value(x_image)
        k_image = k_image.view(B, T_image, self.n_head, self.n_embd // self.n_head).transpose(1, 2) 
        v_image = v_image.view(B, T_image, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        #print(k_image.shape, v_image.shape) # [16, 12, 197, 64]) ([16, 12, 197, 64])
        #print(k_image.transpose(-2, -1).shape)
        att = (q_text @ k_image.transpose(-2, -1)) * (1.0 / math.sqrt(k_image.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T_text, :T_text] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        #print(att.shape) # ([16, 12, 64, 197])
        #print((att @ v_image).transpose(1, 2).shape) #([16, 64, 12, 64])
        return self.c_proj_cross((att @ v_image).transpose(1, 2).contiguous().view(B, T_text, self.n_embd)), att

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
    
class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.dropout_1 = nn.Dropout(cfg.dropout)
        self.dropout_2 = nn.Dropout(cfg.dropout)
        self.dropout_3 = nn.Dropout(cfg.dropout)
        if cfg.peft_type == "adapter":
            self.adapter_layer_1 = AdapterLayer(cfg.n_embd, cfg.adapter_size)
            #self.adapter_layer_2 = AdapterLayer(cfg.n_embd, cfg.adapter_size)
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
        if self.cfg.peft_type == "adapter":
            x = x + self.adapter_layer_1(self.dropout_1(self.attn(self.ln_1(x))))
            x_cross, atten_map = self.attn_cross(self.ln_2(x), x_image)
            x = x + self.dropout_2(x_cross)
            x = x + self.adapter_layer_3(self.dropout_3(self.mlp(self.ln_3(x))))
        elif self.cfg.peft_type == "lora":
            x = x + self.dropout_1(self.attn(self.ln_1(x)))
            x_cross, atten_map = self.attn_cross(self.ln_2(x), x_image)
            x = x + self.dropout_2(x_cross)
            x = x + self.dropout_3(self.mlp(self.ln_3(x)))
        else:
            x = x + self.dropout_1(self.attn(self.ln_1(x)))
            x_cross, atten_map = self.attn_cross(self.ln_2(x), x_image)
            x = x + self.dropout_2(x_cross)
            x = x + self.dropout_3(self.mlp(self.ln_3(x)))
        return x, atten_map
    
class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            # h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            # h = n_layer(self.cfg),
            h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        # layer_name = 'h.0.attn.c_attn'
        # for name, param in self.transformer.named_parameters():
        #     print(name)
        #     if name == layer_name + '.weight':
        #         print(f"Layer: {name}")
        #         print(param.data)
        #         break
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
        atten_map_all = []
        for block in self.transformer.h:
            x, atten_map = block(x, x_image)
            atten_map_all.append(atten_map)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        atten_map_all = torch.stack(atten_map_all)
        # x = self.transformer.h(x, x_image)
        x = self.lm_head(self.transformer.ln_f(x))
        if self.cfg.atten_map_flag:
            return x, atten_map_all
        else:
            return x
    
class PrefixEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
class DecoderPrefixTuning(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # self.prefix_config = [
        #     peft_type="PREFIX_TUNING",
        #     task_type="SEQ_2_SEQ_LM",
        #     num_virtual_tokens=20,
        #     token_dim=768,
        #     num_transformer_submodules=1,
        #     num_attention_heads=12,
        #     num_layers=12,
        #     encoder_hidden_size=768,
        #     prefix_projection=True,
        #     inference_mode=False,
        # ]
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.prefix_encoder = PrefixEncoder(prefix_config)  # Add PrefixEncoder
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.ModuleList([Adapter(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                #print(key)
                if any(key.endswith(w) for w in transposed):
                    #print(key)
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
            
    def forward(self, x, x_image):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        
        # Add PrefixEncoder to encode the prefix
        prefix = torch.randint(0, 20, (x.size(0), 20))
        prefix_embedding = self.prefix_encoder(prefix)
        x = x + prefix_embedding
        
        for block in self.transformer.h:
            x = block(x, x_image)
        x = self.lm_head(self.transformer.ln_f(x))
        return x