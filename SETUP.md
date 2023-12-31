## Instructions for easy setup (not working right now for some reason, not using --extra-index-url correctly)
`conda create --name video python=3.8`
`conda activate video`
`git clone https://github.com/zanedurante/video_understanding.git`
`cd video_understanding` 
`pip install -e .`


## Instructions for manual setup
Assumes cuda version 11.3  For different cuda versions, we will need to update the pytorch version as well.


### Create new conda env for python 3.8
`conda create --name video python=3.8`
`conda activate video`

### install pytorch (latest with capability with cuda=11.3)
`pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

### install pytest for testing
`pip install pytest`

### install pytorch lightning
`pip install lightning`
`pip install pytorch-lightning`

### install remaining dependencies in setup.py 
`pip install ftfy, regex, decord, pandas, black, wandb, matplotlib, omegaconf, opencv-python`
`pip install transformers`

### final step: run pytest in the video_understanding directory.  This will run all tests to make sure you are setup correctly. Warnings are ok
`pytest`