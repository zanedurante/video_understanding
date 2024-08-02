from setuptools import setup, find_packages

# Assumes python version 3.8 and cuda 11.3

setup(
    name="video",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        #"torch==1.12.1+cu113",
        #"torchvision==0.13.1+cu113",
        #"torchaudio==0.12.1",
        "pytest",
        "lightning",
        "ftfy",
        "regex",
        "decord",
        "pandas",
        "black",
        "wandb",
        "matplotlib",
        "omegaconf",
        "transformers",
	"timm",
	"einops",
	"nltk",
    ],
    dependency_links=["https://download.pytorch.org/whl/cu113"],
)
