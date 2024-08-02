import torch
import torch.nn as nn
import pytorch_lightning as pl
from video.video_encoders import get_backbone
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from video.utils.config_manager import get_val_from_config
from torchmetrics.classification import MultilabelConfusionMatrix, MulticlassConfusionMatrix
import numpy as np


class Classifier(pl.LightningModule):
    def __init__(
        self,
        config,
        num_classes=101,
        multilabel=False,
        total_num_steps=1e6,
    ):
        super().__init__()
        self.config = config
        self.multilabel = multilabel
        backbone_name = get_val_from_config(config, "model.backbone_name")
        num_frames = get_val_from_config(config, "data.num_frames")
        self.video_backbone = get_backbone(
            backbone_name, num_frames=num_frames
        ).float()  # TODO: Make configurable somehow
        self.head_type = get_val_from_config(config, "model.head", "linear")
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.confusion_matrix = MultilabelConfusionMatrix(num_labels=num_classes) if self.multilabel else MulticlassConfusionMatrix(num_classes=num_classes)
        self.num_frozen_epochs = get_val_from_config(
            config, "trainer.num_frozen_epochs", 1
        )
        self.total_num_epochs = get_val_from_config(config, "trainer.max_epochs", 10)
        self.backbone_lr_multiplier = get_val_from_config(
            config, "trainer.backbone_lr_multiplier", 0.01
        )
        self.lr = get_val_from_config(config, "trainer.lr", 1e-4)
        if self.multilabel:
            self.train_acc = torchmetrics.classification.Accuracy(
                task="multilabel", num_labels=num_classes
            )
            self.val_acc = torchmetrics.classification.Accuracy(
                task="multilabel", num_labels=num_classes
            )
        else:
            self.train_acc = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.val_acc = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=num_classes
            )
        self.head_weight_decay = get_val_from_config(
            config, "model.head_weight_decay", 0.0
        )
        self.backbone_weight_decay = get_val_from_config(
            config, "model.backbone_weight_decay", 0.0
        )
        self.backbone_is_frozen = False
        self.head_dropout = get_val_from_config(config, "model.head_dropout", 0.0)
        self.total_num_steps = total_num_steps

        if self.multilabel:
            self.loss = nn.BCEWithLogitsLoss() # TODO: Ensure labels are being loaded in correctly
        else:
            self.loss = nn.CrossEntropyLoss()

        # Build the classifier head
        if self.head_type == "linear":
            self.head = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(self.video_backbone.get_video_level_embed_dim(), num_classes),
            )
        elif self.head_type == "mlp":
            hidden_dim = get_val_from_config(
                config, "model.head_hidden_dim", 1024
            )  # By default just use video_level_embed_dim
            print("For MLP, using hidden dim:", hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(
                    self.video_backbone.get_video_level_embed_dim(),
                    hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(self.head_dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise NotImplementedError(
                "Classifier head {} not implemented".format(self.head_type)
            )

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self.num_frozen_epochs > 0:
            if not self.backbone_is_frozen:
                self.freeze_backbone()
        elif self.current_epoch == self.num_frozen_epochs:
            if self.backbone_is_frozen:
                self.unfreeze_backbone()

    def forward(self, batch):
        video = batch["video"]
        video_features = self.video_backbone.get_video_level_embeds(video)
        logits = self.head(video_features)
        if self.multilabel:
            logits = torch.sigmoid(logits)
        return logits

    def freeze_backbone(self):
        print("Freezing backbone")
        for param in self.video_backbone.parameters():
            param.requires_grad = False
        # video backbone to float 16 and eval mode
        # self.video_backbone = self.video_backbone.half()
        self.video_backbone.eval()
        self.backbone_is_frozen = True
        return

    def unfreeze_backbone(self):
        print("Unfreezing backbone")
        for param in self.video_backbone.parameters():
            param.requires_grad = True
        # video backbone to float 32 and train mode
        # self.video_backbone = self.video_backbone.float()
        self.backbone_is_frozen = False
        self.video_backbone.train()
        return

    def training_step(self, batch, batch_idx):
        batch_size = batch["video"].shape[0]
        logits = self(batch)
        labels = batch["label"]
        loss = self.loss(logits, labels)
        train_acc = self.train_acc(logits, labels)
        self.log(
            "train_loss", loss, batch_size=batch_size, prog_bar=True, sync_dist=True
        )
        self.log(
            "train_acc",
            train_acc,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["video"].shape[0]
        logits = self(batch)
        preds = torch.argmax(logits, dim=1) if not self.multilabel else (logits > 0.5).int()
        self.confusion_matrix.update(preds, batch["label"].int())
        labels = batch["label"]
        loss = self.loss(logits, labels)
        val_acc = self.val_acc(logits, labels)
        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            val_acc,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self):
        confusion_matrix = np.array2string(self.confusion_matrix.compute().cpu().numpy())
        print("Rows are the actual classes, columns are predicted classes.")
        print(confusion_matrix)
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        backbone_params = [
            p for p in self.video_backbone.parameters() if p.requires_grad
        ]
        classifier_params = [p for p in self.head.parameters() if p.requires_grad]
        # us self.lr for classifier params and self.lr * self.backbone_lr_multiplier for backbone params
        optimizable_params = [
            {
                "params": backbone_params,
                "lr": self.lr * self.backbone_lr_multiplier,
                "weight_decay": self.backbone_weight_decay,
                "name": "backbone",
            },
            {
                "params": classifier_params,
                "lr": self.lr,
                "weight_decay": self.head_weight_decay,
                "name": "head",
            },
        ]
        # TODO: Make these more configurable
        optimizer = torch.optim.Adam(optimizable_params)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.total_num_steps, eta_min=0.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # "epoch" or "step" for per-step updates
                "frequency": 1,  # Update every epoch/step; adjust as needed
            },
        }
