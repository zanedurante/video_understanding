# Simple 1 gpu training example
import pytorch_lightning as pl
from video.modules.classifier import Classifier
from video.datasets.data_module import VideoDataModule
import torch
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.tuner import Tuner
import os
from glob import glob
from omegaconf import OmegaConf
from video.utils.config_manager import get_config, get_num_workers
from video.utils.module_loader import get_model_module, get_data_module_from_config


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--disable_wandb", action="store_true")
    args.add_argument("--find_lr", action="store_true")
    args.add_argument("--deterministic", "-d", action="store_true")
    args.add_argument("--fast_run", type=bool, default=False)
    args.add_argument("--config", "-c", type=str, default="configs/default.yaml")

    # Add wandb sweep args, add to get_wandb_args and merge_wandb_args in config_manager.py to add more
    args.add_argument(
        "--use_sweep", type=bool, default=False
    )  # Whether using wandb sweep
    args.add_argument("--lr", type=float, default=None)
    args.add_argument("--num_frames", type=int, default=None)
    args.add_argument("--backbone_name", type=str, default=None)
    args.add_argument("--batch_size", type=int, default=None)
    args.add_argument("--head", type=str, default=None)
    args.add_argument("--head_dropout", type=float, default=None)
    args.add_argument("--head_weight_decay", type=float, default=None)
    args.add_argument("--backbone_weight_decay", type=float, default=None)
    args.add_argument("--backbone_lr_multiplier", type=float, default=None)
    args.add_argument("--num_frozen_epochs", type=int, default=None)
    args.add_argument("--max_epochs", type=int, default=None)

    args = args.parse_args()
    return args


def main(args):
    # TODO: use the configs set in configs/
    # TODO: Setup wandb sweep like: https://www.youtube.com/watch?v=WZvG6hwxUEw
    use_lr_finder = args.find_lr
    disable_wandb = args.disable_wandb
    fast_run = args.fast_run
    config_path = args.config
    config = get_config(config_path, args)
    is_deterministic = (
        args.deterministic or config.trainer.is_deterministic
    )  # False by default

    module_type = config.model.type  # TODO: Use this to load the correct module
    num_classes = 101  # Can we infer from the dataset(s)?
    dataset_name = "ucf101"  # Load from datasets in configs

    if is_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Using {} workers".format(config.trainer.num_workers))
    print("Using {} precision".format(config.trainer.precision))

    pl.seed_everything(config.trainer.seed, workers=True)

    if fast_run:
        print("Setting float32 matmul precision to high for fast run")
        torch.set_float32_matmul_precision("high")

    # set logger
    logger = pl.loggers.CSVLogger(
        "logs", name=config.logger.short_run_name
    )  # use short run name for csv logger

    if not disable_wandb:
        with open("wandb.key", "r") as file:
            wandb.login(key=file.read().strip())
        wandb.init(
            project="video_understanding",
            config=OmegaConf.to_container(config),
            name=config.logger.run_name,
        )
        # I think this is unnecessary now
        logger = WandbLogger(
            name=config.logger.run_name, project="video_understanding", config=config
        )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        log_every_n_steps=config.logger.log_every_n_steps,
        deterministic=is_deterministic,
    )

    data_module = get_data_module_from_config(config)


    # total steps = steps per epoch * num epochs
    total_num_steps = (
        data_module.get_stats()["num_train_videos"]
        // config.trainer.batch_size
        * config.trainer.max_epochs
    )

    model_module = get_model_module(config.model.type)

    if type(model_module) == Classifier:
        module = model_module(
            config,
            num_classes=data_module.get_stats()["num_classes"],
            total_num_steps=total_num_steps,
        )
    else:
        module = model_module(
            config,
            total_num_steps=total_num_steps,
        )

    if use_lr_finder:  # TODO: Remove temp .ckpt created from lr finder
        print("Running lr finder...")
        trainer = pl.Trainer()
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            module, data_module, min_lr=1e-12, max_lr=1e-0, num_training=200
        )
        print("LR Finder finished")
        print("Suggested learning rate:", lr_finder.suggestion())
        # lr_finder = trainer.lr_find(module, data_module)
        print(
            "If you are using N GPUs on M nodes, multiply lr by N*M (linear scaling rule)"
        )
        print(
            f"Plot saved to {config.logger.visualization_dir}/lr_finder/lr_finder.png"
        )
        fig = lr_finder.plot(suggest=True)
        os.makedirs(f"{config.logger.visualization_dir}/lr_finder", exist_ok=True)
        fig.savefig(f"{config.logger.visualization_dir}/lr_finder/lr_finder.png")
        # find lr finder ckpts in curr dir and remove
        ckpts_files = glob("*.ckpt")
        for file in ckpts_files:
            # if file start with lr, remove
            if file.startswith("lr"):
                os.remove(file)
        exit()

    trainer.fit(module, data_module)

    if not disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)
