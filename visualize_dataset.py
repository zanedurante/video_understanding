import sys

from video.utils.visualizer import visualize_dataset

dataset_name = sys.argv[1]

visualize_dataset(dataset_name, num_samples=16)
