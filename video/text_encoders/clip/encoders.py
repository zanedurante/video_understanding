from video.video_encoders.clip.clip import load as load_clip
from video.video_encoders.clip.clip import tokenize
from video.video_encoders.clip.encoders import CLIP_MODEL_VARIANTS, MODEL_TO_PATH
from video.text_encoders.base_encoder import BaseTextEncoder
import torch


class CLIPTextEncoder(BaseTextEncoder):
    def __init__(
        self,
        clip_model_name,
        pretrained_path=None,
        frozen=False,
        lora=False,
        device=torch.cuda.current_device(),
    ):
        self.clip_model_name = clip_model_name
        self.pretrained_path = pretrained_path
        self.frozen = frozen
        self.lora = lora
        self.device = device
        super().__init__(
            pretrained_path=pretrained_path,
            frozen=frozen,
            lora=lora,
            device=device,
        )
        clip_model = load_clip(clip_model_name, device=device)[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def get_text_embed_dim(self):
        return self.text_projection.shape[-1]

    def encode_text(self, tokenized_text):
        x = self.token_embedding(tokenized_text).type(
            self.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)]
            @ self.text_projection
        )

        return x

    def tokenize_text(self, text_batch):
        # assumes text_batch is a list of strings
        return tokenize(text_batch).to(self.device)

    # For now we assume the text_batch is a list of strings
    def get_text_embeds(self, text_batch):
        tokenized_text = self.tokenize_text(text_batch)
        text_features = self.encode_text(tokenized_text)
        return text_features


def load_clip_text_encoder(
    clip_model_name,
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
    return CLIPTextEncoder(
        clip_model_name=clip_model_name,
        pretrained_path=pretrained_path,
        frozen=frozen,
        lora=lora,
        device=device,
    )
