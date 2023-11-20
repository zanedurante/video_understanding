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
- [ ] Add CLIP implementation (like ViFi-CLIP)
- [ ] Add initial version of trainer with wandb support in train.py
- [ ] Add debug mode and lr_find flag for running train.py
- [ ] Get good performance on UCF101
- [ ] Add VideoMAEv2 model to codebase
- [ ] Add config system (with base config)
- [ ] Add general trainer code
- [ ] Make as a package (pip install -e .)
- [ ] Create next TODOs
