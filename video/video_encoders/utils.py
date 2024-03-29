import torch
from video.video_encoders.clip.encoders import load_clip_backbone
from video.video_encoders.video_mae.encoders import VideoMAEv2Base


"""
Utils for loading video encoders.
"""


def get_backbone(backbone_name, **kwargs):
    """
    Backbone name is defined by <model-type>_<model-name-id>
    """
    model_type = backbone_name.split("_")[0]
    model_name = "_".join(backbone_name.split("_")[1:])
    if model_type == "clip":
        return load_clip_backbone(model_name, **kwargs)
    elif backbone_name == 'VideoMAEv2Base':
        return VideoMAEv2Base(**kwargs)
    else:
        raise NotImplementedError("Backbone {} not implemented".format(backbone_name))
