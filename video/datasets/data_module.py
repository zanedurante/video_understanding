import pytorch_lightning as pl
from torch.utils.data import DataLoader
from video.datasets.dataset import VideoDataset

# num_classes for classification (if applicable, 0 otherwise). num_train_videos is the number of clips in the training set.
DATASETS_TO_STATS = {
    "ucf101": {
        "num_classes": 101,
        "num_train_videos": 9537,
    },
    "ucf101_wrong": # Created for a sanity check, the first two balance beam videos are re-captioned as baby crawling instead.
    {
        "num_classes": 101,
        "num_train_videos": 9537,
    },
    "ucf101_2":{
        "num_classes": 101,
        "num_train_videos": 9586,
    },
    "ucf101_3":{
        "num_classes": 101,
        "num_train_videos": 9624,
    },
    "500p_0": {
        "num_classes": 0,
        "num_train_videos": 1731,
    },
    "500p_1": {
        "num_classes": 0,
        "num_train_videos": 1273,
    },
    "500p_2": {
        "num_classes": 0,
        "num_train_videos": 1603,
    },
    "500p_new0": {
        "num_classes": 0,
        "num_train_videos": 1731,
    },
    "500p_new1": {
        "num_classes": 0,
        "num_train_videos": 1273,
    },
    "500p_new2": {
        "num_classes": 0,
        "num_train_videos": 1603,
    },
    "webvid_rewritten": {
        "num_classes": 0,
        "num_train_videos": 10716266,
    },
    "webvid_small-rewritten": {
        "num_classes": 0,
        "num_train_videos": 99999,
    },
}


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset_name: str, batch_size: int = 32, num_workers: int = 8, **kwargs
    ):
        super().__init__()
        if dataset_name not in DATASETS_TO_STATS.keys():
            raise NotImplementedError(
                "Dataset {} not implemented.".format(dataset_name)
            )
        if "_" not in dataset_name:
            self.dataset_name = dataset_name
            self.dataset_variant = None
            self.variant_suffix = ""
        else:
            self.dataset_name, self.dataset_variant = dataset_name.split("_")
            self.variant_suffix = "_" + self.dataset_variant
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(
            self.dataset_name,
            dataset_split="train" + self.variant_suffix,
            **self.kwargs
        )
        self.val_dataset = VideoDataset(
            self.dataset_name, dataset_split="val" + self.variant_suffix, **self.kwargs
        )
        self.test_dataset = VideoDataset(
            self.dataset_name, dataset_split="test" + self.variant_suffix, **self.kwargs
        )
        # print("IN SETUP:", len(self.train_dataset))

    def get_stats(self):
        return DATASETS_TO_STATS[self.dataset_name + self.variant_suffix]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
