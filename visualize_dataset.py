import sys
import os
from video.utils.visualizer import visualize_dataset

dataset_name = sys.argv[1]
output_dir = "/home/durante/code/data_viewer/static/grid_videos" #'visualizations' #
if len(sys.argv) > 2:
    output_dir = sys.argv[2]

print("Saving visualizations to", output_dir)

os.makedirs(output_dir, exist_ok=True)

visualize_dataset(dataset_name, output_dir=output_dir, num_samples=-1, num_frames=16, overlay=False)
