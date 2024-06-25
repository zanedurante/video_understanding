import pandas as pd
import os
from tqdm import tqdm
import json

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


def get_category_json_path_from_video_path(video_path):
    category = video_path.split("/")[0]
    # find json that ends with "field.json"
    json_paths = []
    for root, dirs, files in os.walk(os.path.join(DATASET_DIR, category)):
        for file in files:
            if file.endswith("field.json"):
                json_paths.append(os.path.join(root, file))

    if len(json_paths) == 0:
        print("WARNING: No json found for video path {}".format(video_path))

    # choose most recent one, in format like: 2023_06_17_20_42_43_185_field.json
    # (yyyy_mm_dd_hh_mm_ss_ms_field.json)
    json_path = sorted(json_paths)[-1]

    return json_path


def get_val_from_json(json_path, string):
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    val = str(json_dict[string])
    return val


def score2label(rass_score):
    # converts string from format "-5 - Unarousable" to integer label -5
    # format is always "int - string"
    if rass_score == "Not Sure":
        int_val = 0
    else:
        int_val = int(rass_score.split(" ")[0])
    return int_val + 5  # normalize to be 0-9 instead of -5 to 4


category_json_paths = [
    get_category_json_path_from_video_path(path) for path in video_paths
]
print("Getting rass scores")
rass_scores = [get_val_from_json(path, "rass_score") for path in category_json_paths]
print("Getting bed angles")
bed_angles = [get_val_from_json(path, "bed_angle") for path in category_json_paths]

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
    train_rass_scores = []
    train_bed_angles = []
    train_labels = []
    val_test_video_paths = []
    val_test_start_times = []
    val_test_end_times = []
    val_test_captions = []
    val_test_rass_scores = []
    val_test_bed_angles = []
    val_test_labels = []

    for idx, narration_folder in tqdm(
        enumerate(narration_folders), total=len(narration_folders)
    ):
        video_path = video_paths[idx]
        start_time = start_times[idx]
        end_time = end_times[idx]
        caption = captions[idx]
        rass_score = rass_scores[idx]
        bed_angle = bed_angles[idx]
        if narration_folder in train_narration_folders:
            train_video_paths.append(video_path)
            train_start_times.append(start_time)
            train_end_times.append(end_time)
            train_captions.append(caption)
            train_rass_scores.append(rass_score)
            train_bed_angles.append(bed_angle)
            train_labels.append(score2label(rass_score))
        else:
            val_test_video_paths.append(video_path)
            val_test_start_times.append(start_time)
            val_test_end_times.append(end_time)
            val_test_captions.append(caption)
            val_test_rass_scores.append(rass_score)
            val_test_bed_angles.append(bed_angle)
            val_test_labels.append(score2label(rass_score))

    # create train/val+test dataframes
    train_df = pd.DataFrame(
        {
            "video_path": train_video_paths,
            "start_frame": train_start_times,
            "end_frame": train_end_times,
            "caption": train_captions,
            "num_skip_frames": skip_frames,
            "rass_score": train_rass_scores,
            "bed_angle": train_bed_angles,
            "label": train_labels,
        }
    )

    val_df = pd.DataFrame(
        {
            "video_path": val_test_video_paths,
            "start_frame": val_test_start_times,
            "end_frame": val_test_end_times,
            "caption": val_test_captions,
            "num_skip_frames": skip_frames,
            "rass_score": val_test_rass_scores,
            "bed_angle": val_test_bed_angles,
            "label": val_test_labels,
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
        "rass_score": rass_scores,
        "bed_angle": bed_angles,
        "label": [score2label(rass_score) for rass_score in rass_scores],
    }
)

df.to_csv("train_new.csv", index=False)
