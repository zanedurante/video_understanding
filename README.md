# video_understanding
A deep learning library built for video understanding tasks.  Primarily relies upon PyTorch Lightning and wandb.  Takes inspiration from fast.ai 

## Installation:
Clone the repository, then do: `cd video_understanding` and `pip install -e .`.

## Style guides:
Use black to format the code: run `black .` in the main directory.  If you do not have `black` installed, install it with `pip install black`.

## Should you use this repo?
In the current state, absolutely not.  You can see all the TODOs at the bottom of this README.

## TODOs
- [x] Add UCF101 dataset to start
- [ ] Create video frame loader
- [ ] Create video data visualizer (from dataset name)
- [ ] Add config system (with base config)
- [ ] Add CLIP implementation (ViFi-CLIP)
- [ ] Add general trainer code
- [ ] Make as a package (pip install -e .)
- [ ] Create next TODOs
