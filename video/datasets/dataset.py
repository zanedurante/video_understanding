import torch
import pandas as pd
from torch.utils.data import Dataset
import os
from torchvision import transforms

from video.utils.video_reader import load_video

CLIP_PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_PIXEL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_PIXEL_STD = (0.229, 0.224, 0.225)
# TODO: Add mae pixel mean and std


def has_val_split(dataset_name):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(curr_dir, dataset_name, "dataset_dir.txt"), "r") as f:
        dataset_dir = f.read().strip()
    return "val.csv" in os.listdir(dataset_dir)


def get_dataset_dir(dataset_name):
    # Load the dataset dir from <dataset_name>/dataset_dir.txt
    dataset_dir = ""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_dir, dataset_name, "dataset_dir.txt"), "r") as f:
        dataset_dir = f.read().strip()
    return dataset_dir


def get_dataset_csv(dataset_name, dataset_split):
    # Load the dataset dir from <dataset_name>/dataset_dir.txt
    dataset_dir = ""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_dir, dataset_name, "dataset_dir.txt"), "r") as f:
        dataset_dir = f.read().strip()

    if "val" in dataset_split:
        if not has_val_split(dataset_name):
            dataset_split = dataset_split.replace("val", "test")
            print(
                "WARNING: No val split found for dataset {}. Using test split instead.".format(
                    dataset_name
                )
            )

    # Load the dataset csv from <dataset_dir>/<dataset_split>.csv
    csv_path = os.path.join(dataset_dir, dataset_split + ".csv")
    print("Loading dataset csv from {}".format(csv_path))
    df = pd.read_csv(csv_path)
    return df


def init_transform_dict(
    input_res=224,
    center_crop=256,
    randcrop_scale=(0.5, 1.0),
    color_jitter=(0, 0, 0),
    norm_mean=IMAGENET_PIXEL_MEAN,
    norm_std=IMAGENET_PIXEL_STD,
    use_clip_norm=True,
    **kwargs,
):
    # Use normalization from: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L83
    if use_clip_norm:
        norm_mean = CLIP_PIXEL_MEAN
        norm_std = CLIP_PIXEL_STD

    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
                # transforms.RandomHorizontalFlip(), # Disable horizontal flip so that left/right can be used for training.
                transforms.ColorJitter(
                    brightness=color_jitter[0],
                    saturation=color_jitter[1],
                    hue=color_jitter[2],
                ),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(center_crop),
                transforms.CenterCrop(center_crop),
                transforms.Resize(input_res),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(center_crop),
                transforms.CenterCrop(center_crop),
                transforms.Resize(input_res),
                normalize,
            ]
        ),
    }
    return tsfm_dict


def get_transforms(split, **kwargs):
    if "train" in split:
        return init_transform_dict(**kwargs)["train"]
    elif "val" in split:
        return init_transform_dict(**kwargs)["val"]
    elif "test" in split:
        return init_transform_dict(**kwargs)["test"]
    else:
        raise ValueError("Split {} not supported.".format(split))


class VideoDataset(Dataset):
    def __init__(self, dataset_name, dataset_split, num_frames=4, multilabel=False, labels=None, video_path_col="video_path", full_path=False, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.dataset_dir_path = get_dataset_dir(dataset_name)
        self.data = get_dataset_csv(dataset_name, dataset_split)
        self.num_frames = num_frames
        self.transforms = get_transforms(dataset_split, **kwargs)
        self.label_columns = labels # If label_columns have special names
        self.multilabel = multilabel
        self.video_path_col = video_path_col
        self.full_path = full_path # Whether or not the csv contains the full path


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rel_video_path = row[self.video_path_col]
        if self.full_path:
            full_video_path = rel_video_path
        else:
            full_video_path = os.path.join(self.dataset_dir_path, rel_video_path)
        if self.label_columns is None:
            label = row.get("label", 0)  # default to class 0 if label is not present
        else:
            label = row[self.label_columns].tolist()
            if len(label) == 1: # if only one label, convert to scalar (not multilabel)
                label = label[0] 
        caption = row.get(
            "caption", ""
        )  # default to empty string if caption is not present
        question = row.get("question", "")
        answer = row.get("answer", "")
        start_frame = row.get("start_frame", 0)
        end_frame = row.get("end_frame", -1)
        num_skip_frames = row.get("num_skip_frames", -1)
        
        video_tensor, loaded_correctly = load_video(
            full_video_path,
            num_frames=self.num_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            num_skip_frames=num_skip_frames,
            split=self.dataset_split,
        )

        video_tensor = self.transforms(video_tensor)
        if self.multilabel:
            label = torch.tensor(label).float()
        else:
            label = torch.tensor(label).long()

        if not loaded_correctly:
            print("WARNING: Video {} failed to load correctly.".format(full_video_path))
            caption = "A black screen."
            question = "What is occurring in this video?"
            answer = "A black screen."
            if self.multilabel:
                # make labels all 0s for len(self.label_columns) times
                label = torch.zeros(len(self.label_columns)).float()
            else:
                label = torch.tensor(0).long()  # have 0 be the label for black screen

        sample = {
            "video": video_tensor,
            "label": label,
            "caption": caption,
            "question": question,
            "answer": answer,
        }
        return sample

    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
