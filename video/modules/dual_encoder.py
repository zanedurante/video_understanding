import torch
import torch.nn as nn
import pytorch_lightning as pl
from video.video_encoders import get_backbone
from video.text_encoders import get_text_encoder
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR


class DualEncoder(pl.LightningModule):
    def __init__(
        self,
        backbone_name,
        text_encoder_name,
        num_frames,
        head=None,  # By default, just use the heads on the encoders, no additional
        lr=1e-6,
        num_frozen_epochs=0,  # For training the head only
        total_num_epochs=10,
        total_num_steps=1e6,
        backbone_lr_multiplier=0.01,
        backbone_weight_decay=0.0,
        text_encoder_lr_multiplier=0.01,
        text_encoder_weight_decay=0.0,
        head_weight_decay=0.0,
        head_dropout=0.0,
        drop_repeat_text=False,
        shared_embed_dim=None,  # If None, use the default embed dim of the video encoder
    ):
        super().__init__()
        # TODO: Make precision configurable somehow
        self.video_backbone = get_backbone(backbone_name, num_frames=num_frames).float()

        self.text_encoder = get_text_encoder(text_encoder_name).float()
        self.head_type = head
        self.num_frames = num_frames
        self.num_frozen_epochs = num_frozen_epochs
        self.total_num_epochs = total_num_epochs
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.text_encoder_lr_multiplier = text_encoder_lr_multiplier
        self.lr = lr
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass")
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass")
        self.head_weight_decay = head_weight_decay
        self.backbone_weight_decay = backbone_weight_decay
        self.text_encoder_weight_decay = text_encoder_weight_decay
        self.backbone_is_frozen = False
        self.text_encoder_is_frozen = False
        self.head_dropout = head_dropout
        self.total_num_steps = total_num_steps
        self.shared_embed_dim = shared_embed_dim
        self.drop_repeat_text = drop_repeat_text
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1 / 0.07)]))
        # init from visual encoder if it has logit_scale
        if hasattr(self.video_backbone, "logit_scale"):
            print("Initializing logit scale from visual encoder")
            self.logit_scale.data = self.video_backbone.logit_scale.data

        if self.shared_embed_dim is None:
            self.shared_embed_dim = self.video_backbone.get_video_level_embed_dim()

        if head is None:
            self.visual_head = nn.Identity()
            self.text_head = nn.Identity()
        elif head == "linear":
            self.visual_head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(
                    self.video_backbone.get_video_level_embed_dim(),
                    self.shared_embed_dim,
                ),
            )
            self.text_head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(
                    self.text_encoder.get_text_embed_dim(), self.shared_embed_dim
                ),
            )
        elif head == "mlp":
            self.visual_head = nn.Sequential(
                nn.Linear(
                    self.video_backbone.get_video_level_embed_dim(),
                    self.video_backbone.get_video_level_embed_dim(),
                ),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(
                    self.video_backbone.get_video_level_embed_dim(),
                    self.shared_embed_dim,
                ),  # By default just repeat the video_level_embed_dim
            )
            self.text_head = nn.Sequential(
                nn.Linear(
                    self.text_encoder.get_text_embed_dim(),
                    self.text_encoder.get_text_embed_dim(),
                ),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(
                    self.text_encoder.get_text_embed_dim(), self.shared_embed_dim
                ),  # By default just repeat the video_level_embed_dim
            )
        else:
            raise ValueError(f"Invalid head type: {head}")

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self.num_frozen_epochs > 0:
            if not self.backbone_is_frozen:
                self.freeze_backbone()
            if not self.text_encoder_is_frozen:
                self.freeze_text_encoder()
        elif self.current_epoch == self.num_frozen_epochs:
            if self.backbone_is_frozen:
                self.unfreeze_backbone()
            if self.text_encoder_is_frozen:
                self.unfreeze_text_encoder()

    def freeze_backbone(self):
        for param in self.video_backbone.parameters():
            param.requires_grad = False
        self.video_backbone.eval()
        self.backbone_is_frozen = True
        return

    def freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        self.text_encoder_is_frozen = True
        return

    def unfreeze_backbone(self):
        for param in self.video_backbone.parameters():
            param.requires_grad = True
        self.backbone_is_frozen = False
        self.video_backbone.train()
        return

    def unfreeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        self.text_encoder_is_frozen = False
        self.text_encoder.train()
        return

    def forward(self, batch):
        video_features = self.video_backbone.get_video_level_embeds(video)
        text_features = self.text_encoder.encode_text(text)

        video_features = self.visual_head(video_features)
        text_features = self.text_head(text_features)

        # normalized features
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # TODO: Add all_gather for text_features and video_features
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ video_features.t()

        return logits_per_video, logits_per_text

    def training_step(self, batch, batch_idx):
        video = batch["video"]
        text = batch["caption"]
        if self.drop_repeat_text:  # TODO: Move to training step
            good_indices = self.get_text_indices(text)
            video = video[good_indices]
            text = [text[i] for i in good_indices]
        batch_size = len(text)
        logits_per_video, logits_per_text = self({"video": video, "caption": text})
        labels = torch.arange(batch_size).to(logits_per_video.device)
        loss = nn.CrossEntropyLoss()(logits_per_video, labels)
        train_acc = self.train_acc(logits_per_video, labels)
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log(
            "train_acc", train_acc, batch_size=batch_size, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits_per_video, logits_per_text = self(batch)

        labels = torch.arange(batch_size).to(logits_per_video.device)

        # TODO: Modify val metrics to support self.drop_repeat_text and multiple correct answers
        # TODO: Get global accuracy across all val batches

        loss = nn.CrossEntropyLoss()(logits_per_video, labels)
        val_acc = self.val_acc(logits_per_video, labels)
        self.log("val_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)
        self.log(
            "val_acc", val_acc, batch_size=batch_size, on_step=False, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        backbone_params = [
            p for p in self.video_backbone.parameters() if p.requires_grad
        ]
        visual_head_params = [
            p for p in self.visual_head.parameters() if p.requires_grad
        ]
        text_encoder_params = [
            p for p in self.text_encoder.parameters() if p.requires_grad
        ]
        text_head_params = [p for p in self.text_head.parameters() if p.requires_grad]

        optimizable_params = [
            {
                "params": backbone_params,
                "lr": self.lr * self.backbone_lr_multiplier,
                "weight_decay": self.backbone_weight_decay,
                "name": "backbone",
            },
            {
                "params": visual_head_params + text_head_params,
                "lr": self.lr,
                "weight_decay": self.head_weight_decay,
                "name": "head",
            },
            {
                "params": text_encoder_params,
                "lr": self.lr * self.text_encoder_lr_multiplier,
                "weight_decay": self.text_encoder_weight_decay,
                "name": "text_encoder",
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
                "frequency": 1,
            },
        }


# TODO: Currently only drops exact repeats, maybe we should drop similar text?
def get_text_indices(text):
    # Find any indices where the text appears only 1 time
    texts_seen = set()
    good_indices = []
    for idx, t in enumerate(text):
        if t in texts_seen:
            continue
        else:
            texts_seen.add(t)
            good_indices.append(idx)
    return torch.tensor(good_indices)  # torch.tensor(good_indices)
