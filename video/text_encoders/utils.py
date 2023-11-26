import torch
from video.text_encoders.clip.encoders import load_clip_text_encoder

"""
Utils for loading text encoders.
"""


def get_text_encoder(text_encoder_name, **kwargs):
    """
    Text encoder name is defined by <model-type>_<model-name-id>
    """
    model_type = text_encoder_name.split("_")[0]
    model_name = "_".join(text_encoder_name.split("_")[1:])
    if model_type == "clip":
        return load_clip_text_encoder(model_name, **kwargs)
    else:
        raise NotImplementedError(
            "Text encoder {} not implemented".format(text_encoder_name)
        )
