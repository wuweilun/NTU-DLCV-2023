"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

import json
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")

args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True,
        # bnb_4bit_compute_dtype=torch.float16,
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()

# generate caption
caption_dict = {}

# TODO: modify the path
frame_dir = "/home/ai2lab/Desktop/DLCV-Fall-2023-Final-1-catchingstar/data/Charades_frame/frames_fps1"

save_path = "short_hint.json"
# caption_dict = json.load(open(save_path, 'r'))

not_exist_list = []

vid_hint_dict = json.load(open("hint_vid.json", 'r'))

for cnt, need_frame in enumerate(tqdm(vid_hint_dict.keys())):

    video_id, start = need_frame.split("_")
    start = int(start)
    
    image_path = os.path.join(frame_dir, video_id, f"frame_{start:04d}.jpg")
    # check if image exists
    if not os.path.exists(image_path):
        print(f"Image not exists: {image_path}")
        not_exist_list.append((need_frame, image_path))
        continue
    image = Image.open(image_path).convert('RGB')

    old_query = "Describe the image and the activity happened."
    response = vid_hint_dict[need_frame]
    history = [(old_query, response)]
    print(history)

    query = "Describe it shortly."

    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

    # add any transformers params here.
    gen_kwargs = {"max_length": 2048,
                    "do_sample": False} # "temperature": 0.9
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]
        # caption_dict[question_id] = response
        caption_dict[need_frame] = response
        print("\nCog:", response)

    if cnt % 50 == 0:
        with open(save_path, 'w') as fp:
            fp.write(json.dumps(caption_dict))
            print("Successfully save json at cnt: {}".format(cnt))
        
# save json
with open(save_path, 'w') as fp:
    fp.write(json.dumps(caption_dict))
    print("Successfully save json at cnt: {}".format(cnt))

for tup in not_exist_list:
    print(tup)



# ============== template code ==============
        
# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# while True:
#     image_path = input("image path >>>>> ")
#     if image_path == '':
#         print('You did not enter image path, the following will be a plain text conversation.')
#         image = None
#         text_only_first_query = True    
#     else:
#         image = Image.open(image_path).convert('RGB')
    
#     history = []

#     while True:
#         query = input("Human:")
#         if query == "clear":
#             break

#         if image is None:
#             if text_only_first_query:
#                 query = text_only_template.format(query)
#                 text_only_first_query = False
#             else:
#                 old_prompt = ''
#                 for _, (old_query, response) in enumerate(history):
#                     old_prompt += old_query + " " + response + "\n"
#                 query = old_prompt + "USER: {} ASSISTANT:".format(query)

#         if image is None:
#             input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, template_version='base')
#         else:
#             input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

#         inputs = {
#             'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#             'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#             'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#             'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
#         }
#         if 'cross_images' in input_by_model and input_by_model['cross_images']:
#             inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

#         # add any transformers params here.
#         gen_kwargs = {"max_length": 2048,
#                       "do_sample": False} # "temperature": 0.9
#         with torch.no_grad():
#             outputs = model.generate(**inputs, **gen_kwargs)
#             outputs = outputs[:, inputs['input_ids'].shape[1]:]
#             response = tokenizer.decode(outputs[0])
#             response = response.split("</s>")[0]
#             print("\nCog:", response)
#         history.append((query, response))
