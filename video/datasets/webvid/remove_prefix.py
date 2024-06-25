# copied from /mnt/datasets_mnt/webvid10m/metadata and then run this script
import pandas as pd

train_path = "rewritten_train.csv"
test_path = "rewritten_test.csv"

prefix = "/mnt/datasets_mnt/webvid10m/videos/"
len_prefix = len(prefix)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# remove prefix from "video" col, rename col to "video_path"
train_df["video_path"] = train_df["video"].apply(lambda x: x[len_prefix:])
train_df = train_df.drop(columns=["video"])

test_df["video_path"] = test_df["video"].apply(lambda x: x[len_prefix:])
test_df = test_df.drop(columns=["video"])

train_df.to_csv("rewritten_train.csv", index=False)
test_df.to_csv("rewritten_test.csv", index=False)

# save prefix to dataset_dir.txt
with open("dataset_dir.txt", "w") as f:
    f.write(prefix)
