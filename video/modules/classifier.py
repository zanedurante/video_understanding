import torch
import torch.nn as nn
import pytorch_lightning as pl

# TODO: Change to be this import instead:
# from video.video_encoders import get_backbone
from video.video_encoders import get_backbone
import torchmetrics


class Classifier(pl.LightningModule):
    def __init__(
        self,
        backbone_name,
        num_frames,
        num_classes,
        head="linear",
        lr=1e-6,
        num_frozen_epochs=1,
        backbone_lr_multiplier=0.01,
        head_weight_decay=0.0,
        backbone_weight_decay=0.0,
        head_dropout=0.0,
    ):
        super().__init__()
        self.video_backbone = get_backbone(
            backbone_name, num_frames=num_frames
        ).float()  # TODO: Make configurable somehow
        self.head_type = head
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.num_frozen_epochs = num_frozen_epochs
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.lr = lr
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.head_weight_decay = head_weight_decay
        self.backbone_weight_decay = backbone_weight_decay
        self.backbone_is_frozen = False
        self.head_dropout = head_dropout

        # Build the classifier head
        if head == "linear":
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(self.video_backbone.get_video_level_embed_dim(), num_classes),
            )
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(
                    self.video_backbone.get_video_level_embed_dim(),
                    self.video_backbone.get_video_level_embed_dim(),
                ),
                nn.GELU(),
                nn.Dropout(head_dropout),  # TODO: Too high?
                nn.Linear(
                    self.video_backbone.get_video_level_embed_dim(), num_classes
                ),  # By default just repeat the video_level_embed_dim
            )
        else:
            raise NotImplementedError("Classifier head {} not implemented".format(head))

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
        loss = nn.CrossEntropyLoss()(logits, labels)
        train_acc = self.train_acc(logits, labels)
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log(
            "train_acc", train_acc, batch_size=batch_size, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["video"].shape[0]
        logits = self(batch)
        labels = batch["label"]
        loss = nn.CrossEntropyLoss()(logits, labels)
        val_acc = self.val_acc(logits, labels)
        self.log("val_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)
        self.log(
            "val_acc", val_acc, batch_size=batch_size, on_step=False, on_epoch=True
        )
        return loss

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
        return torch.optim.Adam(optimizable_params)
