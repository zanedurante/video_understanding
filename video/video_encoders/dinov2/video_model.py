from video.video_encoders.base_backbone import BaseBackbone
import torch.nn as nn
import torch

DINOV2_MODEL_VARIANTS = [
    "ViT-S/14",
    "ViT-B/14",
    "ViT-L/14",
    "ViT-G/14",
]

MODEL_TO_PATH = {
    "ViT-S/14": "dinov2_vits14",
    "ViT-B/14": "dinov2_vitb14",
    "ViT-L/14": "dinov2_vitl14",
    "ViT-G/14": "dinov2_vitg14",
}


class Dinov2VideoBackbone(BaseBackbone):
    """
    Video encoder for dinov2, encodes each video frame independently using the Dinov2 model.
    Implements the same interface as the other video encoders.
    Specifically:
    - `self.get_spatio_temporal_embeds(video_batch)` generally, these embeddings are structured to match a `(b, t, s, d)` format, representing the batch size, number of frames, and spatio-temporal dimension, and embedding dimension.
    - `self.get_spatio_temporal_embed_dims()` returns a tuple of size `(t, s, d)`.
    - `self.convert_spatio_temporal_embeds_to_video(spatio_temporal_embeds)` returns an embedding of size `(b, d)` representing the batch size and video-level embedding dimension.
    - `self.get_video_level_embeds(video_batch)` gets an embedding of size `(b, d)`, representing the batch size and the video-level embedding dimension.  This calls `self.get_spatio_temporal_embeds` and then `self.convert_spatio_temporal_embeds_to_video`.
    - `self.get_video_level_embed_dim()` returns `d`.
    """

    def __init__(
        self,
        dinov2_model_name,
        num_frames=8,
        pretrained_path=None,
        frozen=False,
        lora=False,
        device=torch.cuda.current_device(),
    ):
        self.dinov2_model_name = dinov2_model_name
        self.num_frames = num_frames
        self.pretrained_path = pretrained_path
        self.frozen = frozen
        self.lora = lora
        self.device = device
        super().__init__(
            pretrained_path=pretrained_path,
            frozen=frozen,
            lora=lora,
            device=device,
            num_frames=num_frames,
        )
        self.dino_model = torch.hub.load('facebookresearch/dinov2', MODEL_TO_PATH[dinov2_model_name], pretrained=True)
        self.video_level_embed_dim = self.dino_model.norm.normalized_shape[0]
        self.spatio_temporal_embed_dims = (self.num_frames,)

    def get_spatio_temporal_embed_dims(self, drop_cls=False):
        # 1 is added to the middle number (50, 197, etc.) for the CLS token
        embed_dims = (0, 0, 0)
        if self.clip_model_name == "ViT-B/32":
            embed_dims = (self.num_frames, 50, 768)
        elif self.clip_model_name == "ViT-B/16":
            embed_dims = (self.num_frames, 197, 768)
        elif self.clip_model_name == "ViT-L/14":
            embed_dims = (self.num_frames, 257, 1024)
        elif self.clip_model_name == "ViT-L/14@336px":
            embed_dims = (self.num_frames, 577, 1024)
        else:
            raise NotImplementedError(
                f"Unsupported CLIP model name {self.clip_model_name}"
            )
        if drop_cls:
            embed_dims = (embed_dims[0], embed_dims[1] - 1, embed_dims[2])
        return embed_dims

    def get_video_level_embed_dim(self):
        return self.video_level_embed_dim

    def get_spatio_temporal_embeds(
        self, video_batch, before_last_block=True, drop_cls=False
    ):
        raise NotImplementedError

    def convert_spatio_temporal_embeds_to_video(self, x, before_last_block=True):
       raise NotImplementedError

    def get_video_level_embeds(self, video_batch):
        # video_batch is (b, t, c, h, w)
        b, t, c, h, w = video_batch.shape
        video_batch = video_batch.reshape(b * t, c, h, w)

        # get the DinoV2 embeddings for each frame individually
        output_embed = self.dino_model(video_batch)
        output_embed_shape = output_embed.shape

        # reshape to (b, t, d)
        frame_embeds = output_embed.reshape(
            b, -1, self.video_level_embed_dim
        )

        # average over temporal dim
        video_level_embeds = frame_embeds.mean(dim=1)
        return video_level_embeds


def load_dinov2_backbone(
    dinov2_model_name,
    num_frames=16,
    pretrained_path=None,
    frozen=False,
    lora=False,
    device="current",
):
    """
    - `pretrained_path=None` Path or URL to the pre-trained video encoder.
    - `frozen=False` Whether or not to freeze the video encoder.
    - `lora=False` Whether or not to fine-tune with LoRA.  `lora` and `frozen` are mutually exclusive.
    - `device='cuda'` Device to load the model on.
    - `num_frames=16` The number of frames to use as input to the model.
    """
    if dinov2_model_name not in DINOV2_MODEL_VARIANTS:
        raise ValueError(
            f"Invalid DINOv2 model name {dinov2_model_name}. Must be one of {DINOV2_MODEL_VARIANTS}"
        )

    if device == "current":
        device = torch.cuda.current_device()
    return Dinov2VideoBackbone(
        dinov2_model_name=dinov2_model_name,
        num_frames=num_frames,
        pretrained_path=pretrained_path,
        frozen=frozen,
        lora=lora,
        device=device,
    )


if __name__ == "__main__":
    model, preprocess = load_clip("ViT-B/32")
    # print(model)
    # print(model.visual)
    # print(preprocess)
    print("Done")
