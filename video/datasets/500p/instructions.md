1. If you are making new splits, might as well redo the entire process since it relies on the old splits (using metadata.csv)
2. Run old_generate_splits.py
3. `sudo cp *0.csv /data/video_narration/; sudo cp *1.csv /data/video_narration/; sudo cp *2.csv /data/video_narration/`
4. update values in `video/datasets/data_module.py`

For new splits:
1. Re-annotate the data by running python app.py in code/data_viewer
2. `cp new.csv /home/durante/code/video_understanding/video/datasets/500p`
3. in this dir: `python generate_csvs.py`
4. (optional) add Q/A to split 0 `python add_qa.py`
4. `sudo cp *0.csv /data/video_narration/; sudo cp *1.csv /data/video_narration/; sudo cp *2.csv /data/video_narration/`