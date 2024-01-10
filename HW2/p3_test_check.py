import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

ground_truth_df = pd.read_csv('./hw2_data/digits/usps/val.csv') 
predictions_df = pd.read_csv('./bash_output/test_usps_pred.csv')  

merged_df = pd.merge(ground_truth_df, predictions_df, left_on='image_name', right_on='image_name', suffixes=('_truth', '_prediction'))

correct_predictions = (merged_df['label_truth'] == merged_df['label_prediction']).sum()

total_samples = len(merged_df)
print(f'USPS samples: {total_samples}')
accuracy = correct_predictions / total_samples

print(f'USPS Accuracy: {accuracy * 100:.2f}%')

ground_truth_df = pd.read_csv('./hw2_data/digits/svhn/val.csv') 
predictions_df = pd.read_csv('./bash_output/test_svhn_pred.csv')  

merged_df = pd.merge(ground_truth_df, predictions_df, left_on='image_name', right_on='image_name', suffixes=('_truth', '_prediction'))

correct_predictions = (merged_df['label_truth'] == merged_df['label_prediction']).sum()

total_samples = len(merged_df)
print(f'SVHN samples: {total_samples}')
accuracy = correct_predictions / total_samples

print(f'SVHN Accuracy: {accuracy * 100:.2f}%')