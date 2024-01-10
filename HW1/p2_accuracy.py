import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

ground_truth_df = pd.read_csv('./hw1_data/p2_data/office/val.csv') 
predictions_df = pd.read_csv('./bash_output/p2_pred.csv')  

merged_df = pd.merge(ground_truth_df, predictions_df, left_on='filename', right_on='filename', suffixes=('_truth', '_prediction'))

correct_predictions = (merged_df['label_truth'] == merged_df['label_prediction']).sum()

total_samples = len(merged_df)
#print(total_samples)
accuracy = correct_predictions / total_samples

print(f'Accuracy: {accuracy * 100:.2f}%')