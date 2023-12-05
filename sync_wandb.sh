#!/bin/bash

# Source the Conda setup script
source /opt/conda/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate video

# Run the wandb sync command
wandb sync /home/durante/code/video_understanding

# You can add additional commands here if needed
