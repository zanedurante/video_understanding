import torch


"""
Utils for loading video encoders.
"""


def load_video_encoder(model_name, **kwargs):
    """
    Loads a video encoder by name.  See video_encoders.md for more information.
    """
    if model_name == "clip":
        raise NotImplementedError(
            "CLIP is not yet implemented in video_encoders.utils.load_video_encoder"
        )
    else:
        raise NotImplementedError(
            "Model {} not implemented in video_encoders.utils.load_video_encoder".format(
                model_name
            )
        )
