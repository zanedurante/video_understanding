"""
Utils for handling pytorch lightning modules.
"""

from video.modules import Classifier, DualEncoder, Captioner
from video.datasets.data_module import VideoDataModule


def get_model_module(module_name):
    module_name = module_name.lower()
    if module_name == "classifier":
        return Classifier
    elif module_name == "dual_encoder":
        return DualEncoder
    elif module_name == "captioner":
        return Captioner
    else:
        raise NotImplementedError(
            "Module {} not implemented in video.utils.module_loader's get_model_module()".format(
                module_name
            )
        )


def get_data_module_from_config(config):
    dataset_list = config.data.datasets
    if len(dataset_list) != 1:
        raise NotImplementedError("Only one dataset supported for now.")
    dataset_name = dataset_list[0]
    # Unless set, infer normalization from model backbone
    if not hasattr(config.data, "norm_type"):
        backbone_name = config.model.backbone_name
        if "clip" in backbone_name.lower():
            norm_type = "clip"
        else:
            norm_type = "imagenet"
    else:
        norm_type = config.data.norm_type.lower()

    print("Using {} normalization.".format(norm_type))

    use_clip_norm = True
    if norm_type != "clip":
        use_clip_norm = False

    module = VideoDataModule(
        dataset_name=dataset_name,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        num_frames=config.data.num_frames,
        use_clip_norm=use_clip_norm,
    )
    return module
