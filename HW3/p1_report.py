import os
import clip
import torch
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Load class-to-label mapping from id2label.json
json_path = "hw3_data/p1_data/id2label.json"
with open(json_path, "r") as f:
    id2label = json.load(f)

# Path to the directory validation images
val_dir = "hw3_data/p1_data/val"
with torch.no_grad():    
    text_inputs_1 = torch.cat([clip.tokenize(f"This is a photo of {c}") for c in id2label.values()]).to(device)
    text_inputs_2 = torch.cat([clip.tokenize(f"This is not a photo of {c}") for c in id2label.values()]).to(device)
    text_inputs_3 = torch.cat([clip.tokenize(f"No {c}, no score") for c in id2label.values()]).to(device)
    text_inputs_4 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in id2label.values()]).to(device)
    prompt_list = [text_inputs_1, text_inputs_2, text_inputs_3, text_inputs_4]
    prompt_name_list = ['this_is_a_photo_of.csv', 'this_is_not_a_photo_of.csv', 'no_score.csv', 'a_photo_of_a.csv']

    for text_inputs, prompt_name in zip(prompt_list, prompt_name_list):
        # Store the results and variables for accuracy calculation
        results_list = []
        correct_predictions = 0
        total_images = 0
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Loop through all images in the directory with tqdm for a progress bar
        for image_name in tqdm(os.listdir(val_dir), desc="Processing images"):
            image_path = os.path.join(val_dir, image_name)

            # Extract the label from the filename
            true_label = int(image_name.split("_")[0])  # Assuming filenames are like "label_id.png"

            # Prepare the inputs
            image = Image.open(image_path)
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Calculate features
            image_features = model.encode_image(image_input)

            # Pick the top 1 most similar label for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            value, index = similarity[0].max(dim=-1)
            index = index.item()

            # Append the result to the list without storing true_label
            results_list.append({"filename": image_name, "predicted_label": index})
            
            # Check for correct predictions
            correct_predictions += int(true_label == index)
            total_images += 1
            
            if prompt_name == 'a_photo_of_a.csv':
                top5_probabilities, top5_indices = torch.topk(similarity[0], 5)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Create a subplot with two axes

                # Display the image on the left subplot
                ax1.imshow(np.array(image))
                ax1.axis('off')
                ax1.set_aspect('auto')
                
                # Plot the top 5 probabilities for the current image
                labels = [id2label[str(i)] for i in top5_indices.cpu().numpy()]
                cmap = plt.get_cmap('Pastel1')
                bars = plt.barh(labels, top5_probabilities.detach().cpu().numpy(), color=cmap(np.arange(len(labels))))
                plt.xlabel('Probability')
                plt.gca().invert_yaxis()
                # Place labels in the middle of each bar
                for bar, label in zip(bars, labels):
                    ax2.text(0.005, bar.get_y() + bar.get_height() / 2, f"a photo of a {label}", ha='left', va='center')
                ax2.set_yticklabels([])
                ax2.set_aspect('auto')
                plt.savefig(f"bash_output/p1_images/{image_name.split('.')[0]}_prob_plot.png")
                plt.close()
            torch.cuda.empty_cache()
            del image_features
        del text_features
        # Save the DataFrame to a CSV file
        results_df = pd.DataFrame(results_list)
        csv_path = f"bash_output/{prompt_name}"
        results_df.to_csv(csv_path, index=False)

        # Calculate accuracy
        accuracy = correct_predictions / total_images * 100
        print(f"\nAccuracy {prompt_name}: {accuracy:.2f}%")
