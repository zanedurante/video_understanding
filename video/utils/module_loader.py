"""
Utils for handling pytorch lightning modules.
"""

from video.modules import Classifier, DualEncoder
from video.datasets.data_module import VideoDataModule


def get_model_module(module_name):
    if module_name == "classifier":
        return Classifier
    elif module_name == "dual_encoder":
        return DualEncoder


def get_data_module_from_config(config):
    dataset_list = config.data.datasets
    if len(dataset_list) != 1:
        raise NotImplementedError("Only one dataset supported for now.")
    dataset_name = dataset_list[0]
    module = VideoDataModule(
        dataset_name=dataset_name,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        num_frames=config.data.num_frames,
    )
    return module
