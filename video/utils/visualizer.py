import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image

from video.datasets.dataset import VideoDataset
from video.datasets.dataset import (
    CLIP_PIXEL_MEAN,
    CLIP_PIXEL_STD,
    IMAGENET_PIXEL_MEAN,
    IMAGENET_PIXEL_STD,
)


def _make_grid(video, ncol=4):
    frame_width = video.shape[3]
    frame_height = video.shape[2]
    num_frames = video.shape[0]

    num_rows = int(np.ceil(num_frames / ncol))

    grid = np.zeros((frame_height * num_rows, frame_width * ncol, 3), dtype=np.uint8)
    for i in range(num_frames):
        x = (i % ncol) * frame_width
        y = (i // ncol) * frame_height
        grid[y : y + frame_height, x : x + frame_width, :] = (
            video[i, :, :, :].numpy().transpose(1, 2, 0) * 255
        )

    return grid


def visualize_sample(sample, use_clip_norm=True, overlay=True, use_grid=True):
    """
    Visualizes a sample from the dataset.
    Sample is a dict with keys: 'video': video_tensor, (optional) 'label': label_tensor, (optional) 'caption': caption_string
    """

    video = sample["video"]
    label = sample.get("label", None)
    caption = sample.get("caption", None)

    if not overlay:
        label, caption = None, None

    # video is a tensor of shape (T, C, H, W)
    # label is a tensor of shape (1)
    # caption is a string
    # overlay the video as a grid of T frames
    # if label is not None, display it as text
    # if caption is not None, display it as text

    # undo normalization
    if use_clip_norm:
        norm_mean = CLIP_PIXEL_MEAN
        norm_std = CLIP_PIXEL_STD
    else:
        norm_mean = IMAGENET_PIXEL_MEAN
        norm_std = IMAGENET_PIXEL_STD

    # reverse original normalization for visualization
    video = video * torch.tensor(norm_std).view(3, 1, 1)
    video = video + torch.tensor(norm_mean).view(3, 1, 1)

    if not use_grid:
        # convert into list of frames
        # reshape (c, h, w) --> (h, w, c)
        frames = [
            Image.fromarray((frame.numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
            for frame in video
        ]

        return frames

    grid = _make_grid(video)
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    if label is not None:
        label = label.item()
        cv2.putText(
            grid, str(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    if caption is not None:
        cv2.putText(
            grid, caption, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    combined_image = grid
    return combined_image


def visualize_dataset(
    dataset_name,
    output_dir="visualizations",
    num_samples=16,
    use_clip_norm=True,
    num_frames=4,
    overlay=True,
    use_gif=False,  # saves as gif instead of grid
    shuffle=True,
):
    """
    Visualizes a dataset by loading samples from it and visualizing it.
    """
    # remove after _ if present in dataset name
    if "_" in dataset_name:
        modifier = "_" + dataset_name.split("_")[1]
        dataset_name = dataset_name.split("_")[0]
    else:
        modifier = ""

    dataset = VideoDataset(
        dataset_name, dataset_split="train" + modifier, num_frames=num_frames
    )

    if shuffle:
        dataset.shuffle()

    if num_samples == -1:
        num_samples = len(dataset)
    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        print("Label for", i, sample["label"])
        if use_gif:
            frames = visualize_sample(
                sample, use_clip_norm=use_clip_norm, overlay=overlay, use_grid=False
            )

            # save video as gif
            video_path = "{}/{}.gif".format(output_dir, i)

            frames[0].save(
                video_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
                optimize=False,
            )

        else:
            combined_image = visualize_sample(
                sample, use_clip_norm=use_clip_norm, overlay=overlay
            )
            # save image to visualizations/{i}.png
            cv2.imwrite("{}/{}.png".format(output_dir, i), combined_image)
