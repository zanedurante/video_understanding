import sys

from video.utils.visualizer import visualize_dataset

dataset_name = sys.argv[1]
output_dir = "visualizations"
if len(sys.argv) > 2:
    output_dir = sys.argv[2]

print("Saving visualizations to", output_dir)

visualize_dataset(dataset_name, output_dir=output_dir, num_samples=16)
