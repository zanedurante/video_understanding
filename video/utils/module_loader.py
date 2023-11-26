"""
Utils for handling pytorch lightning modules.
"""

from video.modules import Classifier, DualEncoder


def get_module(module_name):
    if module_name == "classifier":
        return Classifier
    elif module_name == "dual_encoder":
        return DualEncoder
