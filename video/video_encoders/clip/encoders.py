from video.video_encoders.clip.clip import load as load_clip
from video.video_encoders.base_backbone import BaseBackbone
import torch.nn as nn
import torch

CLIP_MODEL_VARIANTS = [
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
]

MODEL_TO_PATH = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


class CLIPVideoBackbone(BaseBackbone):
    """
    Video encoder for clip, encodes each video frame independently using the CLIP model.
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
        clip_model_name,
        num_frames=8,
        pretrained_path=None,
        frozen=False,
        lora=False,
        device=torch.cuda.current_device(),
    ):
        self.clip_model_name = clip_model_name
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
        clip_model = load_clip(clip_model_name)[0]
        self.logit_scale = clip_model.logit_scale
        self.clip_model = clip_model.visual
        self.video_level_embed_dim = self.clip_model.output_dim
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
        # video_batch is (b, t, c, h, w)
        b, t, c, h, w = video_batch.shape
        video_batch = video_batch.reshape(b * t, c, h, w)

        # get the CLIP embeddings (before the last block, )
        video_batch = self.clip_model(video_batch, before_last_block=True)

        if drop_cls:
            video_batch = video_batch[:, 1:, :]

        # output shape
        output_embed_shape = self.get_spatio_temporal_embed_dims(drop_cls=drop_cls)

        return video_batch.reshape(
            b, output_embed_shape[0], output_embed_shape[1], output_embed_shape[2]
        )

    def convert_spatio_temporal_embeds_to_video(self, x, before_last_block=True):
        # spatio_temporal_embeds is (b, t, s, d)
        b, t, s, d = x.shape
        # flatten to (b * t, s, d)
        x = x.reshape(b * t, s, d)

        # run the last block of the encoder + remaining blocks
        if before_last_block:
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer.resblocks[-1](x)

            x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.ln_post(x[:, 0, :])

        if self.clip_model.proj is not None:
            x = x @ self.clip_model.proj

        # Temporal averaging of cls token along the temporal dimension
        x = x.reshape(b, t, self.video_level_embed_dim)

        # average along the temporal dimension
        x = x.mean(dim=1)

        return x

    def get_video_level_embeds(self, video_batch):
        st_embeds = self.get_spatio_temporal_embeds(video_batch)
        video_embeds = self.convert_spatio_temporal_embeds_to_video(st_embeds)
        return video_embeds


def load_clip_backbone(
    clip_model_name,
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
    if clip_model_name not in CLIP_MODEL_VARIANTS:
        raise ValueError(
            f"Invalid CLIP model name {clip_model_name}. Must be one of {CLIP_MODEL_VARIANTS}"
        )
    if "RN" in clip_model_name:
        raise NotImplementedError(
            f"RN models are not yet supported. Please use a ViT model."
        )
    if device == "current":
        device = torch.cuda.current_device()
    return CLIPVideoBackbone(
        clip_model_name=clip_model_name,
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
