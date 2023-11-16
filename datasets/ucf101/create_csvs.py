# Create a csv for each split of the UCF101 dataset

import os
import csv
import sys
from constants import classes

if len(sys.argv) != 2:
    print("Usage: python create_csvs.py <oath/to/dataset_dir>")
    exit(1)

dataset_dir = sys.argv[1]
# save to dataset_dir.txt
with open("dataset_dir.txt", "w") as f:
    f.write(dataset_dir)

split_dir = "ucfTrainTestlist"

class2idx = {}
idx2class = {}

# read in class name tsv
class_name_tsv = os.path.join(split_dir, "classInd.txt")
with open(class_name_tsv, "r") as f:
    class_names = f.readlines()

for index, class_name in enumerate(class_names):
    # split string and take last element as class name
    class_name = class_name.split(" ")[-1].strip()
    class2idx[class_name] = index
    idx2class[index] = class_name


# Paths are in the format: <class_name>/<video_name>.avi
def get_class_name_from_path(path):
    return path.split("/")[0]


# default is split_num=1
def create_csv(split_name="train", split_num=1):
    list_file = os.path.join(
        split_dir, "{}list{:02d}.txt".format(split_name, split_num)
    )

    with open(list_file, "r") as f:
        file_paths = f.readlines()

    # remove newlines
    file_paths = [path.strip().split(" ")[0] for path in file_paths]

    # get class names
    class_names = [get_class_name_from_path(path) for path in file_paths]
    class_indices = [class2idx[class_name] for class_name in class_names]
    clip_class_names = [classes[idx] for idx in class_indices]

    # save in format <video_path>,<class_index>
    if split_num == 1:
        csv_file = os.path.join(dataset_dir, "{}.csv".format(split_name))
    else:
        csv_file = os.path.join(dataset_dir, "{}_{}.csv".format(split_name, split_num))

    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        # create header first
        writer.writerow(["video_path", "label", "caption"])

        # write rest of rows
        writer.writerows(zip(file_paths, class_indices, clip_class_names))


# write csvs for all splits
for split_name in ["train", "test"]:
    for split_num in [1, 2, 3]:
        create_csv(split_name, split_num)
