# Simple 1 gpu training example
import pytorch_lightning as pl
from video.modules.classifier import Classifier
from video.datasets.data_module import VideoDataModule
import torch
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
import pytorch_lightning

print(pytorch_lightning.__version__)
from pytorch_lightning.tuner import Tuner
import os
from glob import glob
from omegaconf import OmegaConf
from video.utils.config_manager import get_config, get_num_workers
from video.utils.module_loader import get_model_module, get_data_module_from_config
from video.utils.confusion_matrix_callback import ConfusionMatrixCallback  
from pytorch_lightning.utilities import rank_zero_only


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
    args.add_argument("--head_hidden_dim", type=int, default=None)
    args.add_argument("--backbone_weight_decay", type=float, default=None)
    args.add_argument("--backbone_lr_multiplier", type=float, default=None)
    args.add_argument("--num_frozen_epochs", type=int, default=None)
    args.add_argument("--max_epochs", type=int, default=None)
    args.add_argument("--text_encoder_weight_decay", type=float, default=None)
    args.add_argument("--text_encoder_lr_multiplier", type=float, default=None)
    args.add_argument("--text_decoder_lr_multiplier", type=float, default=None)
    args.add_argument("--text_decoder_weight_decay", type=float, default=None)
    args.add_argument("--prompt_lr_multiplier", type=float, default=None)
    args.add_argument("--prompt_weight_decay", type=float, default=None)
    args.add_argument("--text_first", type=bool, default=None)
    args.add_argument("--num_learnable_prompt_tokens", type=int, default=None)
    args.add_argument("--use_start_token_for_caption", type=bool, default=None)
    args.add_argument("--prompt", type=str, default=None)
    args.add_argument("--backbone_pretrained_ckpt", type=str, default=None)

    args = args.parse_args()
    return args


def main(args):
    use_lr_finder = args.find_lr
    disable_wandb = args.disable_wandb
    fast_run = args.fast_run
    config_path = args.config
    config = get_config(config_path, args)
    print("Config: {}".format(config))
    is_deterministic = (
        args.deterministic or config.trainer.is_deterministic
    )  # False by default

    if is_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Using {} workers".format(config.trainer.num_workers))
    print("Using {} precision".format(config.trainer.precision))

    pl.seed_everything(config.trainer.seed, workers=True)

    data_module = get_data_module_from_config(config)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath="pytorch_lightning_ckpts/",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )


    # callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(EarlyStopping(monitor="val_loss", patience=3, mode="min"))
    
    if fast_run:
        print("Setting float32 matmul precision to high for fast run")
        torch.set_float32_matmul_precision("high")

    # set logger
    logger = pl.loggers.CSVLogger(
        "logs", name=config.logger.short_run_name
    )  # use short run name for csv logger

    # Only log to wandb from rank 0 process
    if not disable_wandb and rank_zero_only.rank == 0:
        with open("wandb.key", "r") as file:
            wandb.login(key=file.read().strip())
        print("Running wandb init")

        wandb.init(
            project="video_understanding",
            config=OmegaConf.to_container(config),
            name=config.logger.run_name,
        )
        print("Creating wandb logger")
        # I think this is unnecessary now
        logger = WandbLogger(
            name=config.logger.run_name, project="video_understanding", config=config
        )
        print("Finished creating wandb logger")

    print("Logging config for reproducibility: {}".format(config))

    # detect number of devices and use that number
    num_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        devices=num_gpus,
        accelerator="gpu",
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        enable_checkpointing=False, # TODO: Make configurable. Too many checkpoints right now
        callbacks=callbacks,
        log_every_n_steps=config.logger.log_every_n_steps,
        deterministic=is_deterministic,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
    )

    # total steps = steps per epoch * num epochs
    total_num_steps = (
        data_module.get_stats()["num_train_videos"]
        // config.trainer.batch_size
        * config.trainer.max_epochs
    )

    print("Getting model module")
    model_module = get_model_module(config.model.type)
    if model_module is Classifier:
        print("Num classes: {}".format(data_module.get_stats()["num_classes"]))
        module = model_module(
            config,
            num_classes=data_module.get_stats()["num_classes"],
            multilabel=data_module.multilabel,
            total_num_steps=total_num_steps,
        )
    else:
        module = model_module(
            config,
            total_num_steps=total_num_steps,
        )
    print("Loaded model module")

    if use_lr_finder:  # TODO: Remove temp .ckpt created from lr finder
        print("Running lr finder...")
        trainer = pl.Trainer()
        print("Making tuner")
        tuner = Tuner(trainer)
        print("Finding lr...")
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
    if config.trainer.skip_test:
        print("Skipping test")
    else:
        trainer.test(
            module, data_module
        )  # TODO: Change to only trigger if config is set

    if not disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)
