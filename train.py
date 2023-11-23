# Simple 1 gpu training example
import pytorch_lightning as pl
from video.modules.classifier import Classifier
from video.datasets.data_module import VideoDataModule
import torch
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.tuner import Tuner
import os
from glob import glob
from omegaconf import OmegaConf
from video.utils.config_manager import get_config, get_num_workers


def main(args):
    # TODO: use the configs set in configs/
    # TODO: Setup wandb sweep like: https://www.youtube.com/watch?v=WZvG6hwxUEw
    use_lr_finder = args.find_lr
    disable_wandb = args.disable_wandb
    fast_run = args.fast_run
    config_path = args.config
    config = get_config(config_path)
    is_deterministic = args.deterministic or config.trainer.is_deterministic # False by default

    module_type = config.model.type # TODO: Use this to load the correct module
    num_classes = 101 # Can we infer from the dataset(s)?
    dataset_name = "ucf101" # Load from datasets in configs

    if is_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Using {} workers".format(config.trainer.num_workers))
    print("Using {} precision".format(config.trainer.precision))

    pl.seed_everything(config.trainer.seed, workers=True)

    if fast_run:
        print("Setting float32 matmul precision to high for fast run")
        torch.set_float32_matmul_precision('high')

    # set logger
    logger = pl.loggers.CSVLogger("logs", name=config.logger.run_name)
        
    if not disable_wandb:
        with open("wandb.key", "r") as file:
            wandb.login(key=file.read().strip())
        logger = WandbLogger(name=config.logger.run_name, project="video_understanding")
        
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        callbacks = [
            LearningRateMonitor(logging_interval='step', log_momentum=True), # TODO: Find why this is not appearing in wandb logs
            ],
        log_every_n_steps=config.logger.log_every_n_steps,
        deterministic=is_deterministic,
    )

    module = Classifier(
        backbone_name=config.model.backbone_name,
        num_frames=config.data.num_frames,
        num_classes=num_classes,
        head=config.model.head,
        lr=config.trainer.lr,
        num_frozen_epochs=config.trainer.num_frozen_epochs,
        backbone_lr_multiplier=config.trainer.backbone_lr_multiplier,
        head_weight_decay=config.model.head_weight_decay,
        backbone_weight_decay=config.model.backbone_weight_decay,
    )

    data_module = VideoDataModule(
        dataset_name=dataset_name,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        num_frames=config.data.num_frames,
    )

    if use_lr_finder: # TODO: Remove temp .ckpt created from lr finder
        print("Running lr finder...")
        trainer = pl.Trainer()
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(module, data_module, min_lr=1e-12, max_lr=1e-0, num_training=200)
        print("LR Finder finished")
        print("Suggested learning rate:", lr_finder.suggestion())
        #lr_finder = trainer.lr_find(module, data_module)
        print("If you are using N GPUs on M nodes, multiply lr by N*M (linear scaling rule)")
        print(f"Plot saved to {config.logger.visualization_dir}/lr_finder/lr_finder.png")
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
    args = argparse.ArgumentParser()
    args.add_argument("--disable_wandb", action="store_true")
    args.add_argument("--find_lr", action="store_true")
    args.add_argument("--deterministic", "-d", action="store_true")
    args.add_argument("--fast_run", action="store_true")
    args.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    args = args.parse_args()
    main(args)