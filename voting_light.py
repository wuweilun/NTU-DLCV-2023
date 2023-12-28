import json

with open('origin.json', 'r') as file1:
    data1 = json.load(file1)

with open('llama_epoch6.json', 'r') as file2:
    data2 = json.load(file2)

# Get the each type 
interaction_data = data1.get("Interaction", [])
sequence_data = data1.get("Sequence", [])
prediction_data = data1.get("Prediction", [])
feasibility_data = data2.get("Feasibility", [])

# Merge the data
merged_data = {"Interaction": interaction_data, "Sequence": sequence_data, "Prediction": prediction_data, "Feasibility": feasibility_data}

# Write the merged data into a new JSON file
with open('./output_dir/voting_5988.json', 'w') as merged_file:
    json.dump(merged_data, merged_file, indent=2)
