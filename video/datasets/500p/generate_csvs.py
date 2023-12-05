import pandas as pd
import os
from tqdm import tqdm

# use metadata.csv to generate initial splits
df = pd.read_csv("new.csv")

DATASET_DIR = "/data/video_narration/"

# print columns
print(df.columns)

# do initial split of train/val+test by narration folder
# use 67/33 split
start_times = df["start_frame"]
end_times = df["end_frame"]
captions = df["caption"]
global_captions = df["global_caption"]
video_paths = [path.replace(DATASET_DIR, "") for path in df["video_path"]]

# first level of path
narration_folders = [path.split("/")[0] for path in video_paths]

# get unique narration folders
unique_narration_folders = list(set(narration_folders))

# use 4 skip frames to get 3 fps videos
skip_frames = 4

# split into train/val+test, create 3 folds
# use 67/33 split
for i in range(3):
    train_idxs = []
    val_test_idxs = []
    for j, narration_folder in enumerate(unique_narration_folders):
        if j % 3 == i:
            val_test_idxs.append(j)
        else:
            train_idxs.append(j)
    train_narration_folders = set([unique_narration_folders[idx] for idx in train_idxs])
    val_test_narration_folders = set(
        [unique_narration_folders[idx] for idx in val_test_idxs]
    )

    # get video_paths, start_times, end_times, captions for train/val+test
    train_video_paths = []
    train_start_times = []
    train_end_times = []
    train_captions = []
    val_test_video_paths = []
    val_test_start_times = []
    val_test_end_times = []
    val_test_captions = []

    for idx, narration_folder in tqdm(
        enumerate(narration_folders), total=len(narration_folders)
    ):
        video_path = video_paths[idx]
        start_time = start_times[idx]
        end_time = end_times[idx]
        caption = captions[idx]
        if narration_folder in train_narration_folders:
            train_video_paths.append(video_path)
            train_start_times.append(start_time)
            train_end_times.append(end_time)
            train_captions.append(caption)
        else:
            val_test_video_paths.append(video_path)
            val_test_start_times.append(start_time)
            val_test_end_times.append(end_time)
            val_test_captions.append(caption)

    # create train/val+test dataframes
    train_df = pd.DataFrame(
        {
            "video_path": train_video_paths,
            "start_frame": train_start_times,
            "end_frame": train_end_times,
            "caption": train_captions,
            "num_skip_frames": skip_frames,
        }
    )

    val_df = pd.DataFrame(
        {
            "video_path": val_test_video_paths,
            "start_frame": val_test_start_times,
            "end_frame": val_test_end_times,
            "caption": val_test_captions,
            "num_skip_frames": skip_frames,
        }
    )

    # save train/val+test dataframes
    train_df.to_csv(os.path.join("train_new{}.csv".format(i)), index=False)
    val_df.to_csv(os.path.join("test_new{}.csv".format(i)), index=False)

# Create train.csv with all data
df = pd.DataFrame(
    {
        "video_path": video_paths,
        "start_frame": start_times,
        "end_frame": end_times,
        "caption": captions,
        "num_skip_frames": skip_frames,
        "global_caption": global_captions,
    }
)

df.to_csv("train_new.csv", index=False)
