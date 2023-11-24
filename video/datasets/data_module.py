import pytorch_lightning as pl
from torch.utils.data import DataLoader
from video.datasets.dataset import VideoDataset

IMPLEMENTED_DATASETS = ["ucf101"]

DATASETS_TO_STATS = {
    "ucf101": {
        "num_classes": 101,
        "num_train_videos": 9537,
    }
}


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset_name: str, batch_size: int = 32, num_workers: int = 8, **kwargs
    ):
        super().__init__()
        if dataset_name not in IMPLEMENTED_DATASETS:
            raise NotImplementedError(
                "Dataset {} not implemented.".format(dataset_name)
            )
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(
            self.dataset_name, dataset_split="train", **self.kwargs
        )
        self.val_dataset = VideoDataset(
            self.dataset_name, dataset_split="val", **self.kwargs
        )
        self.test_dataset = VideoDataset(
            self.dataset_name, dataset_split="test", **self.kwargs
        )
        # print("IN SETUP:", len(self.train_dataset))

    def get_stats(self):
        return DATASETS_TO_STATS[self.dataset_name]

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
