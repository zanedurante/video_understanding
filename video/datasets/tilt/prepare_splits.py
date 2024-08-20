from glob import glob
import pandas as pd

test_videos_class_0 = glob('/home/e/mobile/tilt_30/dataset5/test/horizont/*.jpg')
test_videos_class_1 = glob('/home/e/mobile/tilt_30/dataset5/test/tilt_horizont/*.jpg')

train_videos_class_0 = glob('/home/e/mobile/tilt_30/dataset5/train/horizont/*.jpg')
train_videos_class_1 = glob('/home/e/mobile/tilt_30/dataset5/train/tilt_horizont/*.jpg')


with open('dataset_dir.txt', 'r') as f:
    dataset_dir = f.read()

column_names = "image_path,class"
train_df = pd.DataFrame(columns=column_names.split(','))
test_df = pd.DataFrame(columns=column_names.split(','))

train_df['image_path'] = train_videos_class_0 + train_videos_class_1
train_df['class'] = [0]*len(train_videos_class_0) + [1]*len(train_videos_class_1)

test_df['image_path'] = test_videos_class_0 + test_videos_class_1
test_df['class'] = [0]*len(test_videos_class_0) + [1]*len(test_videos_class_1)

train_df.to_csv(dataset_dir + '/train.csv', index=False)
test_df.to_csv(dataset_dir + '/val.csv', index=False)
test_df.to_csv(dataset_dir + '/test.csv', index=False)

print("Number in train: ", len(train_df))
