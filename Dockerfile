# Build: docker build -t <image_name:version> .
# The script was created following the SETUP.md instructions
#FROM pytorch/pytorch:1.13.0-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
#
#RUN conda install python=3.8
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install pytest
RUN pip install lightning
RUN pip install regex
RUN pip install decord
RUN pip install pandas
RUN pip install black
RUN pip install wandb
RUN pip install matplotlib
RUN pip install omegaconf
RUN pip install opencv-python
RUN pip install ftfy
RUN pip install transformers 
#RUN pip install pytorch-lightning==2.1.2