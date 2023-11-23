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

def main(args):
    # TODO: use the configs set in configs/
    # TODO: Setup wandb sweep like: https://www.youtube.com/watch?v=WZvG6hwxUEw
    use_lr_finder = args.find_lr
    debug_mode = args.debug
    disable_wandb = args.disable_wandb
    fast_run = args.fast_run
    visualization_dir = "visualizations"
    backbone_name = "clip_ViT-B/16"
    num_frames = 16
    num_classes = 101 # Can we infer from the dataset?
    classifier_head = "mlp"
    dataset_name = "ucf101"
    batch_size = 16
    num_workers = os.cpu_count() - 2 #8
    max_epochs = 10
    precision = 32 # set to 16?
    lr = 1e-2
    num_frozen_epochs = 2
    backbone_lr_multiplier = 0.00001
    log_every_n_steps=10 # default is 50
    seed=42
    is_deterministic=False
    classifier_head_weight_decay = 0.001
    backbone_weight_decay = 0.0

    if is_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Using {} workers".format(num_workers))
    print("Using {} precision".format(precision))

    pl.seed_everything(seed, workers=True)

    if fast_run:
        print("Setting float32 matmul precision to high for fast run")
        torch.set_float32_matmul_precision('high')
    
    #if use_lr_finder:
    #    num_frozen_epochs = 0 # frozen epochs make LR finder not work properly, maybe?

    # set logger
    run_name = f"{backbone_lr_multiplier}lr-mult_{backbone_name}_{num_frames}frames_{classifier_head}class-head_{dataset_name}"
    logger = pl.loggers.CSVLogger("logs", name=run_name)
        
    if not disable_wandb:
        with open("wandb.key", "r") as file:
            wandb.login(key=file.read().strip())
        logger = WandbLogger(name=run_name, project="video_understanding")
        
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        precision=precision,
        max_epochs=max_epochs,
        logger=logger,
        callbacks = [
            LearningRateMonitor(logging_interval='step', log_momentum=True), # TODO: Find why this is not appearing in wandb logs
            ],
        log_every_n_steps=log_every_n_steps,
        deterministic=is_deterministic,
    )

    module = Classifier(
        backbone_name=backbone_name,
        num_frames=num_frames,
        num_classes=num_classes,
        classifier_head=classifier_head,
        lr=lr,
        num_frozen_epochs=num_frozen_epochs,
        backbone_lr_multiplier=backbone_lr_multiplier,
        classifier_head_weight_decay=classifier_head_weight_decay,
        backbone_weight_decay=backbone_weight_decay,
    )

    data_module = VideoDataModule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        num_frames=num_frames,
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
        print(f"Plot saved to {visualization_dir}/lr_finder/lr_finder.png")
        fig = lr_finder.plot(suggest=True)
        os.makedirs(f"{visualization_dir}/lr_finder", exist_ok=True)
        fig.savefig(f"{visualization_dir}/lr_finder/lr_finder.png")
        exit()


    trainer.fit(module, data_module)

    if not disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--disable_wandb", action="store_true")
    args.add_argument("--find_lr", action="store_true")
    args.add_argument("--debug", action="store_true")
    args.add_argument("--fast_run", action="store_true")
    args = args.parse_args()
    main(args)