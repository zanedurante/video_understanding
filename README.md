# video_understanding
A deep learning library built for video understanding tasks.  Primarily relies upon PyTorch Lightning and wandb.  Takes inspiration from fast.ai 

## Installation:
Clone the repository, then do: `cd video_understanding` and `pip install -e .`.

## Style guides:
Use black to format the code: run `black .` in the main directory.  If you do not have `black` installed, install it with `pip install black`.

## Before submitting commits
Format with `black .` and run `pytest` in the parent directory.

## Should you use this repo?
In the current state, absolutely not.  You can see all the TODOs at the bottom of this README.

## TODOs
- [x] Add UCF101 dataset to start
- [x] Create video frame loader
- [x] Create video data visualizer (from dataset name)
- [x] Add CLIP implementation (like ViFi-CLIP)
- [x] Add initial version of trainer with wandb support in train.py
- [x] Add general trainer code
- [x] Add debug mode and lr_find flag for running train.py
- [x] Get good classification performance on UCF101
- [x] Make as a package (pip install -e .)
- [x] Delete temp ckpt file created
- [x] Figure out why the learning rate and momentum is not logged to wandb
- [x] Create good nested config system
- [ ] Fix config test
- [ ] Add hyperparameter sweeps with wandb
- [ ] Allow for multiple datasets for training (how to weight?)
- [ ] Detect the number of classes for the dataset(s) automatically
- [ ] Extend codebase to support video-text matching as a task.
- [ ] Add multiple dataset support + Kinetics-400
- [ ] Add NTP part to the codebase
- [ ] Add the webvid rewritten dataset to the codebase.
- [ ] Add VideoMAEv2 model to codebase? Maybe we just do a frame-level MAE model instead?
- [ ] Add LoRA fine-tuning capabilities (especially important for LLaMA models and maybe for video encoders too)
- [ ] Create next TODOs
