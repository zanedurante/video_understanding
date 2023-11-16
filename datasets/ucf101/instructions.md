Download the dataset from the official source: 
```
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
```

Use unrar to decompress the files to `UCF-101` directory:
`unrar x UCF101.rar`

Download the official train-test splits from here: https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip in this directory.

unzip it in this directory (ucf101/ucfTrainTestlist)

Create csv files with:
`python create_csvs.py /path/to/UCF-101`

