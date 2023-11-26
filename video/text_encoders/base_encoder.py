import torch
import torch.nn as nn

"""
Implements the base class for all backbones.  Functionality of this class is in video_encoders.md

Each backbone has the following functionality:
- `self.get_spatio_temporal_embeds(video_batch)` generally, these embeddings are structured to match a `(b, t, s, d)` format, representing the batch size, number of frames, and spatio-temporal dimension, and embedding dimension.
- `self.get_spatio_temporal_embed_dims()` returns a tuple of size `(t, s, d)`.
- `self.convert_spatio_temporal_embeds_to_video(spatio_temporal_embeds)` returns an embedding of size `(b, d)` representing the batch size and video-level embedding dimension.
- `self.get_video_level_embeds(video_batch)` gets an embedding of size `(b, d)`, representing the batch size and the video-level embedding dimension.  This calls `self.get_spatio_temporal_embeds` and then `self.convert_spatio_temporal_embeds_to_video`.
- `self.get_video_level_embed_dim()` returns `d`.

Base backbone args (to be used for all backbones in the load_<backbone-name>_backbone functions)
- `pretrained_path=None` Path or URL to the pre-trained video encoder.
- `frozen=False` Whether or not to freeze the video encoder.
- `lora=False` Whether or not to fine-tune with LoRA.  `lora` and `frozen` are mutually exclusive.
- `device='current'` Device to load the model on. if 'current' uses torch.cuda.current_device()
- `num_frames=16` The number of frames to use as input to the model.

"""


class BaseTextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_path=None,
        frozen=False,
        lora=False,
        device="cuda",
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.frozen = frozen
        self.lora = lora
        self.device = device
        print("Loading model to {}".format(device))
        self.to(device)

    def _load_pretrained(self):
        if self.pretrained_path is not None:
            print("Loading pretrained model from {}".format(self.pretrained_path))
            # Load to cpu first
            ckpt_dict = self.load_state_dict(
                torch.load(self.pretrained_path, map_location=torch.device("cpu"))
            )
            print("ckpt_dict:", ckpt_dict)
        if self.frozen:
            print("Freezing backbone model")
            for param in self.parameters():
                param.requires_grad = False

    def get_text_embeds(self, text_batch):
        raise NotImplementedError

    def get_text_embed_dim(self):
        raise NotImplementedError
