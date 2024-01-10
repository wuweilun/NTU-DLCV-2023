# prompt: https://github.com/openai/CLIP/blob/main/data/prompts.md#cifar100
# zeroshot_classifier: https://github.com/openai/CLIP/blob/fcab8b6eb92af684e7ff0a904464be7b99b49b88/notebooks/Prompt_Engineering_for_ImageNet.ipynb

import os
import clip
import torch
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd
import sys

# Only turn on when I test the accuracy
accuracy_flag = False

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# Load images, id2label.json, csv_path
image_dir = sys.argv[1]
json_path = sys.argv[2]
csv_path = sys.argv[3]

with open(json_path, "r") as f:
    id2label = json.load(f)

templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Processing texts"):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


zeroshot_weights = zeroshot_classifier(id2label.values(), templates)

# Store the results and variables for accuracy calculation
results_list = []
correct_predictions = 0
total_images = 0

# Loop through all images in the directory with tqdm for a progress bar
for image_name in tqdm(os.listdir(image_dir), desc="Processing images"):
    image_path = os.path.join(image_dir, image_name)

    # Prepare the inputs
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Pick the top 1 most similar label for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100. * image_features @ zeroshot_weights).softmax(dim=-1)
    value, index = similarity[0].max(dim=-1)
    index = index.item()

    # Append the result to the list without storing true_label
    results_list.append({"filename": image_name, "label": index})
    
    if(accuracy_flag):
        # Extract the label from the filename
        true_label = int(image_name.split("_")[0])  # Assuming filenames are like "label_id.png"
        
        # Check for correct predictions
        correct_predictions += int(true_label == index)
        total_images += 1

# Save the DataFrame to a CSV file
results_df = pd.DataFrame(results_list)
results_df.to_csv(csv_path, index=False)

if(accuracy_flag):
    # Calculate accuracy
    accuracy = correct_predictions / total_images * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
