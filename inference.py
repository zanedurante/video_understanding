import pytorch_lightning as pl
from video.modules.classifier import Classifier
from video.utils.video_reader import _load_image
from video.datasets.dataset import get_transforms
from omegaconf import OmegaConf
import torch

## NEED TO PRELOAD THIS PART

config = OmegaConf.load("/home/e/clip/datasets/video_understanding/normal_tilt_ckpts/config.yaml")

class_dict = {
    0: "normal", # valid
    1: "tilt", # invalid
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(class_dict)

transforms = get_transforms("test")

model = Classifier.load_from_checkpoint("/home/e/clip/datasets/video_understanding/normal_tilt_ckpts/epoch=8-val_loss=0.23.ckpt", config=config, num_classes=num_classes)
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
    input_tensor = transforms(_load_image(input_path, 1, "test")[0]).to(device)
    output = get_output(input_tensor)
    return output
    #return class_dict[output]



if __name__ == "__main__":
    example_input = "/home/e/mobile/tilt_30/dataset5/test/horizont/a1ea5b0b-861d-4eb8-b91e-5a26fbee6161.jpg"

    print(get_output_from_path(example_input))
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("/home/e/mobile/tilt_30/dataset5/test.csv")
    paths = df["image_path"].values
    values = df["class"].values

    preds = []
    for path in tqdm(paths):
        preds.append(get_output_from_path(path))
    
    acc = (values == preds).mean()
    print("Accuracy: ", acc)