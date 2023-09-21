import pandas as pd

ground_truth_df = pd.read_csv('./hw1_data/p1_data/val_gt.csv') 
predictions_df = pd.read_csv('./bash_output/p1_pred.csv')  

merged_df = pd.merge(ground_truth_df, predictions_df, left_on='image_id', right_on='filename', suffixes=('_truth', '_prediction'))

correct_predictions = (merged_df['label_truth'] == merged_df['label_prediction']).sum()

total_samples = len(merged_df)

accuracy = correct_predictions / total_samples

print(f'Accuracy: {accuracy * 100:.2f}%')