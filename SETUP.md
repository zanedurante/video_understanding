## Instructions for easy setup (linux)
TODO: Make setup script, currently not implemented

## Instructions for manual setup
Assumes cuda version 11.3  For different cuda versions, we will need to update the pytorch version as well.

For a snapshot of a fully working environment as of October 18, 2024 see [requirements.txt](requirements.txt).  The following instructions use all of the environments specified there.  For our specific install, we faced the same issue reported on the PyTorch GitHub [here](https://github.com/pytorch/pytorch/issues/111469). To fix it, we added `export LD_LIBRARY_PATH=/opt/conda/envs/video/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH` to the end of the `./bashrc` file.  You should replace `/opt/conda/envs/video/` with the path to your anaconda environment for this package.

### Create new conda env for python 3.8
`conda create --name video python=3.11`

`conda activate video`

### install pytorch (latest with capability with cuda=11.3)
`pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0`

### install pytest for testing
`pip install pytest`

### install pytorch lightning
`pip install lightning==2.4.0`

### install remaining dependencies
`pip install ftfy regex decord pandas black wandb matplotlib omegaconf opencv-python`

`pip install transformers==4.45.2`

`pip install timm einops scikit-learn`

### git submodule setup
In the main `video_understanding` directory, do:
`git submodule init`

`git submodule update`

`pip install -e .`


### (optional): Setup crontab for wandb syncing (useful for if wandb cannot connect normally)
`crontab -e`

Paste at the bottom of the file: `*/5 * * * * /home/durante/code/video_understanding/sync_wandb.sh`

### final step: run pytest in the video_understanding directory.  This will run all tests to make sure you are setup correctly. Warnings are ok -- should take about 3-5 mins to run
`pytest`
