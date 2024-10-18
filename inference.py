import pytorch_lightning as pl
from video.modules.classifier import Classifier
from video.utils.video_reader import load_video
from video.datasets.dataset import get_transforms
from omegaconf import OmegaConf
import torch

## NEED TO PRELOAD THIS PART

config = OmegaConf.load("configs/classification/icu_movement.yaml")

class_dict = {
    0: "None", # valid
    1: "Assisted Movement", # invalid
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(class_dict)

transforms = get_transforms("test")

model = Classifier.load_from_checkpoint("/share/pi/schul/durante/icu_movement_ckpts/epoch=5-val_loss=0.12.ckpt", config=config, num_classes=num_classes)
model = model.to(device)
model.eval()

def get_output(input_tensor, threshold=0.5):
    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0) # Add dimension for batch
        features = model.video_backbone.get_video_level_embeds(input_tensor)
        logits = model.head(features)
        preds = logits > threshold
        return preds.int().argmax().item()


### RUN THIS PART WITH API
def get_output_from_path(input_path):
    input_tensor = transforms(load_video(input_path, 1, "test")[0]).to(device)
    output = get_output(input_tensor)
    return output
    #return class_dict[output]


def get_output_from_1min_path(input_path, fps=15):
    # Assuming 1 minute video at 15 fps --> divide into 8 second clips like: 
    # seconds 0 - 8, 8 - 16, 16 - 24, 24 - 32, 32 - 40, 40 - 48, 48 - 56, 56 - 60 (final clip is not sampled exactly correctly)
    # Then take max of all the clips
    start_times = [0, 8, 16, 24, 32, 40, 48, 56]
    end_times = [8, 16, 24, 32, 40, 48, 56, 60]

    start_frames = [int(start_time * fps) for start_time in start_times]
    end_frames = [int(end_time * fps) for end_time in end_times]

    preds = []
    for idx, start_frame, end_frame in enumerate(zip(start_frames, end_frames)):
        video, loaded_correctly = load_video(input_path, num_frames=32, start_frame=start_frame, end_frame=end_frame, split="test")
        if not loaded_correctly:
            preds.append(0)
            continue
        input_tensor = transforms(video).to(device)
        output = get_output(input_tensor)
        preds.append(output)
    
    return max(preds)

if __name__ == "__main__":
    example_input = "/share/pi/schul/icu/sample_1min/2022_01_01_15_00_m440_lt.mp4"
    output = get_output_from_1min_path(example_input)
    print("Model output:", output)

if __name__ == "old__main__":
    example_input = "/share/pi/schul/icu/icu_narration_clips/2023_06_21_trish/clip_05_03/rt_0018.mp4"

    print(get_output_from_path(example_input))
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("/share/pi/schul/icu/test_movement-balanced.csv")
    paths = df["video_path"].values
    values = df["AssistedMovement"].values

    preds = []
    for path in tqdm(paths):
        full_path = "/share/pi/schul/icu/" + path
        preds.append(get_output_from_path(full_path))
    
    acc = (values == preds).mean()
    print("Accuracy: ", acc)

    # Add column to dataframe
    df["preds"] = preds

    # Save dataframe with predictions
    df.to_csv("/share/pi/schul/icu/test_movement-balanced_preds.csv", index=False)