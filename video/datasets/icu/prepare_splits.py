# For now, just create basic 80/20 splits (not split by patient)
# Load in csv from /share/pi/schul/icu/combined_labels.csv

import pandas as pd
import numpy as np

with open("dataset_dir.txt", "r") as file:
    data_dir = file.read().strip()

combined_data = pd.read_csv(data_dir + "/combined_labels.csv")

# Create 80/20 split from indices
np.random.seed(0)
indices = np.random.permutation(combined_data.index)
split = int(0.8 * len(indices))
train_indices = indices[:split]
val_indices = indices[split:]

# Create new train.csv and val.csv
train_data = combined_data.loc[train_indices]
val_data = combined_data.loc[val_indices]

# Change column name from "clip" to "video_path"
train_data.rename(columns={"clip": "video_path"}, inplace=True)
val_data.rename(columns={"clip": "video_path"}, inplace=True)

# Save to files
train_data.to_csv(data_dir + "/train.csv", index=False)
val_data.to_csv(data_dir + "/val.csv", index=False)
val_data.to_csv(data_dir + "/test.csv", index=False) # For now, use val as test since initial run