import pandas as pd
train_orig = pd.read_csv('train_new0.csv')
val_orig = pd.read_csv('test_new0.csv')

train_qa = pd.read_csv('train_new0_qa.csv')
val_qa = pd.read_csv('test_new0_qa.csv')

idxs_train_qa = train_qa['idx'].tolist()
idxs_val_qa = val_qa['idx'].tolist()

train_qs = []
train_as = []
val_qs = []
val_as = []

for idx in range(len(train_orig)):
    if idx not in idxs_train_qa:
        train_qs.append("What is happening in the video?")
        train_as.append(train_orig['caption'][idx])
    else:
        train_qs.append(train_qa[train_qa['idx'] == idx]['question'].values[0])
        train_as.append(train_qa[train_qa['idx'] == idx]['answer'].values[0])

for idx in range(len(val_orig)):
    if idx not in idxs_val_qa:
        val_qs.append("What is happening in the video?")
        val_as.append(val_orig['caption'][idx])
    else:
        val_qs.append(val_qa[val_qa['idx'] == idx]['question'].values[0])
        val_as.append(val_qa[val_qa['idx'] == idx]['answer'].values[0])

train_orig['question'] = train_qs
train_orig['answer'] = train_as
val_orig['question'] = val_qs
val_orig['answer'] = val_as

# save the new dataframes as qa csvs

train_orig.to_csv('train_qa0.csv', index=False)
val_orig.to_csv('test_qa0.csv', index=False)

# Combine the dfs into one big df and create new 80:10:10 splits
combined_df = pd.concat([train_orig, val_orig], ignore_index=True)

# shuffle the rows
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# first .80 for train, next .10 for val, last .10 for test
train_df = combined_df[:int(len(combined_df)*.8)]
val_df = combined_df[int(len(combined_df)*.8):int(len(combined_df)*.9)]
test_df = combined_df[int(len(combined_df)*.9):]

# save as new csvs: train_avl0.csv, val_avl0.csv, test_avl0.csv
train_df.to_csv('train_avl0.csv', index=False)
val_df.to_csv('val_avl0.csv', index=False)
test_df.to_csv('test_avl0.csv', index=False)
        