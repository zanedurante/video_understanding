from omegaconf import OmegaConf
import os

BASE_CONFIG_PATH = "configs/base.yaml"


def get_config(config_path, args=None):
    config_subdir = os.path.dirname(config_path)
    base_config = OmegaConf.load(BASE_CONFIG_PATH)
    subdir_config = OmegaConf.load(BASE_CONFIG_PATH)
    if config_subdir != "configs":
        subdir_config_path = os.path.join(config_subdir, "default.yaml")
        if os.path.exists(subdir_config_path):
            print(
                "Loading default from subdir: {} instead of base.yaml".format(
                    config_subdir
                )
            )
            subdir_config = OmegaConf.load(os.path.join(config_subdir, "default.yaml"))
    config = OmegaConf.merge(base_config, subdir_config)
    config = OmegaConf.merge(config, OmegaConf.load(config_path))
    config.trainer.num_workers = get_num_workers(config.trainer.num_workers)

    # Override with wandb sweep args
    wandb_args = get_wandb_args(args)
    config = merge_wandb_args(config, wandb_args)

    config.logger.run_name, config.logger.short_run_name = get_run_name(config)
    return config


def merge_wandb_args(config, wandb_args):
    # Merge args that are not none, otherwise keep the default
    for key, value in wandb_args.items():
        if key == "lr":
            if value is not None:
                config.trainer.lr = value
        elif key == "num_frames":
            if value is not None:
                config.data.num_frames = value
        elif key == "backbone_name":
            if value is not None:
                config.model.backbone_name = value
        elif key == "batch_size":
            if value is not None:
                config.trainer.batch_size = value
        elif key == "head":
            if value is not None:
                config.model.head = value
        elif key == "head_dropout":
            if value is not None:
                config.model.head_dropout = value
        elif key == "head_weight_decay":
            if value is not None:
                config.trainer.head_weight_decay = value
        elif key == "backbone_weight_decay":
            if value is not None:
                config.trainer.backbone_weight_decay = value
        elif key == "backbone_lr_multiplier":
            if value is not None:
                config.trainer.backbone_lr_multiplier = value
        elif key == "num_frozen_epochs":
            if value is not None:
                config.trainer.num_frozen_epochs = value
        elif key == "max_epochs":
            if value is not None:
                config.trainer.max_epochs = value
        elif key == "text_encoder_weight_decay":
            if value is not None:
                config.trainer.text_encoder_weight_decay = value
        elif key == "text_encoder_lr_multiplier":
            if value is not None:
                config.trainer.text_encoder_lr_multiplier = value
        elif key == "text_decoder_lr_multiplier":
            if value is not None:
                config.trainer.text_decoder_lr_multiplier = value
        elif key == "text_decoder_weight_decay":
            if value is not None:
                config.trainer.text_decoder_weight_decay = value
        elif key == "prompt_lr_multiplier":
            if value is not None:
                config.trainer.prompt_lr_multiplier = value
        elif key == "prompt_weight_decay":
            if value is not None:
                config.trainer.prompt_weight_decay = value
        elif key == "text_first":
            if value is not None:
                config.model.text_first = value
        elif key == "num_learnable_prompt_tokens":
            if value is not None:
                config.model.num_learnable_prompt_tokens = value
        elif key == "use_start_token_for_caption":
            if value is not None:
                config.model.use_start_token_for_caption = value
    return config


def get_wandb_args(args):
    arg_list = [
        "lr",
        "num_frames",
        "backbone_name",
        "batch_size",
        "head",
        "head_dropout",
        "head_weight_decay",
        "backbone_weight_decay",
        "backbone_lr_multiplier",
        "num_frozen_epochs",
        "max_epochs",
        "fast_run",
        "text_encoder_lr_multiplier",
        "text_encoder_weight_decay",
        "text_decoder_lr_multiplier",
        "text_decoder_weight_decay",
        "prompt_lr_multiplier",
        "prompt_weight_decay",
        "text_first",
        "num_learnable_prompt_tokens",
        "use_start_token_for_caption",
    ]
    wandb_args = {}
    if args is None:
        return wandb_args

    for arg_name, arg_value in args.__dict__.items():
        if arg_name in arg_list:
            wandb_args[arg_name] = arg_value
    return wandb_args


def get_num_workers(config_str):
    # format is either cpus-<int>, cpus/<int>, or <int>
    try:
        if config_str.startswith("cpus-"):
            return os.cpu_count() - int(config_str.split("-")[1])
        elif config_str.startswith("cpus/"):
            return os.cpu_count() // int(config_str.split("/")[1])
        else:
            return int(config_str)
    except:
        raise Warning(
            "Invalid num_workers in config: {}. Needs to be an int or expression relative to number of cpus like: cpus-2 or cpus/4. Defaulting to 1.".format(
                config_str
            )
        )
    return 1


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
        if key not in priority_keys and key not in ignored_keys and key in config.keys()
    ]

    run_name = "_".join(priority_parts + remaining_parts)
    short_run_name = run_name[: min(len(run_name), 100)]
    return run_name, short_run_name


def get_val_from_config(config, key, default=None):
    keys = key.split(".")
    val = config
    for key in keys:
        if key in val:
            val = val[key]
        else:
            return default

    return val
