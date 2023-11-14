## General information
All datasets used in this repo use the same format:

```
video_path,caption/label,(optional)start_frame,(optional)end_frame,(optional)num_skip_frames
```

We assume each dataset is stored in `DATASET_DIR/dataset_name`, where `dataset_name` is the name of the dataset directory. 
`DATASET_DIR` is a constant defined in constants.py.  Feel free to use symlinks to make this easier.  Thus, the path to each video is: `DATASET_DIR/dataset_name/video_path`.

### Contents of csv file
If `caption` is used, we assume it is a video-text pair dataset.  If `label` is used, we assume it is a classification dataset.  
These are the only types of datasets supported so far.

`start_frame` and `end_frame` are the start and end frames within the video located in `video_path`.  `start_frame` defaults to 0, and `end_frame` defaults to -1 (last frame).

`num_skip_frames` is by default set to `-1`, meaning it is not used.  If set to `N > 0`, we will skip every `N` frames during sampling.  This is used to control fps.  For example, `num_skip_frames`  set to `0`, it will not skip any frames (use raw video fps).  If `num_skip_frames` is set to `1`, we skip every other frame, etc.  By default, we sample randomly throughout the video during test and evenly space throughout the video during evaluation, resuling in uneven fps.  This is fine for many applications but can seriously impact results in others.  It is often dataset-specific, or even video-specific, which is why we specify it here.

### What to do when adding a new dataset
Each dataset should contain a subdirectory within this parent folder.  The dataset will have the following files and components:
1. `instructions.md` for getting the dataset into the format expected (including download source, and any extra code used).
2. `train.csv` for training csv in the format specified above.
3. `val.csv` for a held-out validation set to be used during the run.
4. `(optional) test.csv` for a held-out test set to be used after the run.  This file will only exist for datasets with official test splits.
5. Create additional splits with different settings, for example `train_4fps.csv` to train at 4 fps.  Just add an underscore and then the setting!
