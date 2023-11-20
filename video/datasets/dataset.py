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


def has_val_split(dataset_name):
    with open(dataset_name + "/dataset_dir.txt", "r") as f:
        dataset_dir = f.read().strip()
    return dataset_dir + "/val.csv" in os.listdir(dataset_dir)


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

    if dataset_split == "val":
        if not has_val_split(dataset_name):
            dataset_split = "test"
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


def get_transforms(split):
    if split in ["train", "val", "test"]:
        return init_transform_dict()[split]
    else:
        raise ValueError("Split {} not supported.".format(split))


class VideoDataset(Dataset):
    def __init__(self, dataset_name, dataset_split, num_frames=4):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.dataset_dir_path = get_dataset_dir(dataset_name)
        self.data = get_dataset_csv(dataset_name, dataset_split)
        self.num_frames = num_frames
        self.transforms = get_transforms(dataset_split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rel_video_path = row["video_path"]
        full_video_path = os.path.join(self.dataset_dir_path, rel_video_path)
        label = row.get("label", None)
        caption = row.get("caption", None)
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

        if not loaded_correctly:
            print("WARNING: Video {} failed to load correctly.".format(full_video_path))
            caption = "A black screen."

        label = torch.tensor(label).long()

        sample = {"video": video_tensor, "label": label, "caption": caption}
        return sample
