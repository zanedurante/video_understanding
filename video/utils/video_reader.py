import decord
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


decord.bridge.set_bridge("torch")


def load_video(
    video_path,
    num_frames=4,
    start_frame=0,
    end_frame=-1,
    num_skip_frames=-1,
    split="train",
    save_error_videos=False,
):
    """
    Loads a video from a given path and returns a tensor of shape (T, C, H, W).

    """
    # special loading for gif files
    if video_path.endswith(".gif"):  # currently not supported for num_skip_frames < 0
        return _load_video_gif(
            video_path, start_frame, end_frame, num_skip_frames, split
        )

    frame_indices = None
    try:
        video_reader = decord.VideoReader(video_path, num_threads=1)
    except:
        print("Error loading video: {}".format(video_path))
        if save_error_videos:
            print("Saving path to error file...")
            with open("/mnt/datasets_mnt/output/error_videos.txt", "a") as f:
                f.write(video_path + "\n")

        imgs = Image.new("RGB", (224, 224), (0, 0, 0))
        imgs = transforms.ToTensor()(imgs).unsqueeze(0)
        imgs = imgs.repeat(num_frames, 1, 1, 1)
        return imgs, False
    video_length = len(video_reader)

    if (
        num_skip_frames < 0 and split == "train"
    ):  # Use random frame sampling for training
        frame_indices = np.random.randint(0, video_length, num_frames)
        frame_indices = np.sort(frame_indices)
    elif (
        num_skip_frames < 0 and split != "train"
    ):  # Use evenly spread frames for val and testing
        frame_indices = np.linspace(0, video_length - 1, num_frames, dtype=int)
    else:
        # TODO: Implement variable/random offsets for fixed FPS sampling
        frame_indices = np.arange(start_frame, end_frame, num_skip_frames + 1)[
            :num_frames
        ]

    frames = video_reader.get_batch(frame_indices)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, True


def _load_video_gif(
    video_path,
    num_frames=4,
    start_frame=0,
    end_frame=-1,
    num_skip_frames=-1,
    split="train",
):
    to_tensor = transforms.ToTensor()
    try:
        with Image.open(video_path) as img:
            video_length = img.n_frames
            if (
                num_skip_frames < 0 and split == "train"
            ):  # Use random frame sampling for training
                frame_indices = np.random.randint(0, video_length, num_frames)
                frame_indices = np.sort(frame_indices)
            elif (
                num_skip_frames < 0 and split != "train"
            ):  # Use evenly spread frames for val and testing
                frame_indices = np.linspace(0, video_length - 1, num_frames, dtype=int)
            else:
                frame_indices = np.arange(start_frame, end_frame, num_skip_frames + 1)[
                    :num_frames
                ]

            frames = []
            for frame_idx in frame_indices:
                img.seek(frame_idx)
                frame = img.convert("RGB")
                frame = to_tensor(frame)
                frames.append(frame)
            return torch.stack(frames), True
    except:  # load black frames
        imgs = Image.new("RGB", (224, 224), (0, 0, 0))
        imgs = transforms.ToTensor()(imgs).unsqueeze(0)
        # Repeat num_frames times in first dim
        imgs = imgs.repeat(num_frames, 1, 1, 1)
        return imgs, False
