from omegaconf import OmegaConf
import os

BASE_CONFIG_PATH = "configs/base.yaml"


def get_config(config_path):
    config_subdir = os.path.dirname(config_path)
    base_config = OmegaConf.load(BASE_CONFIG_PATH)
    subdir_config = OmegaConf.load(BASE_CONFIG_PATH)
    if config_subdir != "configs":
        print("Attempting to load from subdir: {}".format(config_subdir))
        subdir_config_path = os.path.join(config_subdir, "default.yaml")
        if os.path.exists(subdir_config_path):
            print("Loading from subdir: {}".format(config_subdir))
            subdir_config = OmegaConf.load(os.path.join(config_subdir, "default.yaml"))

    config = OmegaConf.merge(base_config, subdir_config)
    config = OmegaConf.merge(config, OmegaConf.load(config_path))
    config.trainer.num_workers = get_num_workers(config.trainer.num_workers)
    config.logger.run_name = get_run_name(config)
    print(config)
    return config


def get_num_workers(config_str):
    # format is either <int> or cpus-<int>
    if config_str.startswith("cpus-"):
        return os.cpu_count() - int(config_str.split("-")[1])
    else:
        return int(config_str)


# use - to separate key and value, use _ to separate key-value pairs
def normalize_key_value(key, value):
    return f"{key}-{value}"


# Get the run name from the config
def get_run_name(config):
    # Set the arguments that are not used in the run name
    priority_keys = ["data", "model"]
    ignored_keys = ["logger"]
    priority_parts = [
        normalize_key_value(key, config[key])
        for key in priority_keys
        if key in config.keys()
    ]

    remaining_parts = [
        normalize_key_value(key, config[key])
        for key in config.keys()
        if key not in priority_keys and key not in ignored_keys
    ]

    run_name = "_".join(priority_parts + remaining_parts)
    return run_name
