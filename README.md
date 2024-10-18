# video_understanding
A deep learning library built for video understanding tasks.  Primarily relies upon PyTorch Lightning and wandb.  Takes inspiration from fast.ai 

## Installation:
Follow the instruction steps in the [Setup Instructions](SETUP.md).


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
- [x] Fix config test
- [x] Add hyperparameter sweeps with wandb
- [x] Detect the number of classes for the dataset(s) automatically (requires setup)
- [x] Finishing touches on wandb sweeps
- [ ] Configure model checkpoint locations
- [ ] Create evaluate.py (give config, create args for metrics, and give test/val.csv)
- [ ] Renew pytest-bed
- [ ] Add hmdb dataset
- [ ] Add Kinetics-400 dataset
- [ ] Add Kinetics-700 dataset
- [ ] Add WebVid10M dataset
- [ ] Add RewrittenWebVid dataset
- [ ] Revisit codebase to support video-text matching as a task.
- [ ] Allow for multiple datasets for training (how to weight?)
- [ ] Modify code to allow for large csv loading
- [ ] Add multiple dataset support + Kinetics-400
- [ ] Add NTP part to the codebase
- [ ] Add the webvid rewritten dataset to the codebase.
- [ ] Add VideoMAEv2 model to codebase? Maybe we just do a frame-level MAE model instead?
- [ ] Add LoRA fine-tuning capabilities (especially important for LLaMA models and maybe for video encoders too)
- [ ] Explore other optimizers?
- [ ] Create next TODOs
