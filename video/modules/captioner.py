import torch
import torch.nn as nn
import pytorch_lightning as pl
from video.video_encoders import get_backbone
from video.text_decoders import get_text_decoder
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from video.utils.config_manager import get_val_from_config
import numpy as np
from transformers import OPTForCausalLM, GPT2Tokenizer

# TODO: Implement LLaMA


class Captioner(pl.LightningModule):
    """
    Basic idea: video encoder and text decoder are trained to predict the next tokens in the caption
    """

    def __init__(
        self,
        config,
        total_num_steps=1e6,
    ):
        super().__init__()
        self.config = config
        backbone_name = get_val_from_config(config, "model.backbone_name")
        self.num_frames = get_val_from_config(config, "data.num_frames")
        text_decoder_name = get_val_from_config(config, "model.text_decoder_name")
        self.video_backbone = get_backbone(
            backbone_name, num_frames=self.num_frames
        ).float()  # TODO: How to make precision configurable
        text_first = get_val_from_config(config, "model.text_first", True)
        num_learnable_prompt_tokens = get_val_from_config(
            config, "model.num_learnable_prompt_tokens", 0
        )
        use_start_token_for_caption = get_val_from_config(
            config, "model.use_start_token_for_caption", False
        )
        self.text_decoder = get_text_decoder(
            text_decoder_name,
            text_first=text_first,
            num_learnable_prompt_tokens=num_learnable_prompt_tokens,
            use_start_token_for_caption=use_start_token_for_caption,
        ).float()  # TODO: Need to figure out how to do different precision
        self.head_type = get_val_from_config(
            config, "model.head"
        )  # For captioner, head is the adaption network
        self.head_dropout = get_val_from_config(config, "model.head_dropout", 0.0)
        self.num_frozen_epochs = get_val_from_config(
            config, "trainer.num_frozen_epochs", 0
        )  # only head is trained
        self.total_num_steps = total_num_steps
        self.total_num_epochs = get_val_from_config(config, "trainer.max_epochs", 10)
        self.backbone_lr_multiplier = get_val_from_config(
            config, "trainer.backbone_lr_multiplier", 1.0
        )
        self.text_decoder_lr_multiplier = get_val_from_config(
            config, "trainer.text_decoder_lr_multiplier", 1.0
        )
        self.prompt_lr_multiplier = get_val_from_config(
            config, "trainer.prompt_lr_multiplier", 1.0
        )
        self.lr = get_val_from_config(config, "trainer.lr", 1e-4)
        self.batch_size = get_val_from_config(config, "trainer.batch_size", 16)
        self.val_batch_size = get_val_from_config(
            config, "trainer.val_batch_size", self.batch_size
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.text_decoder.vocab_size
        )
        self.train_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.text_decoder.vocab_size
        )
        self.val_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.head_weight_decay = get_val_from_config(
            config, "trainer.head_weight_decay", 0.0
        )
        self.backbone_weight_decay = get_val_from_config(
            config, "trainer.backbone_weight_decay", 0.0
        )
        self.text_decoder_weight_decay = get_val_from_config(
            config, "trainer.text_decoder_weight_decay", 0.0
        )
        self.prompt_weight_decay = get_val_from_config(
            config, "trainer.prompt_weight_decay", 0.0
        )
        self.drop_cls_token = get_val_from_config(config, "model.drop_cls_token", True)
        self.backbone_is_frozen = False
        self.text_decoder_is_frozen = False
        vid_embed_dims = self.video_backbone.get_spatio_temporal_embed_dims(
            drop_cls=self.drop_cls_token
        )
        self.num_visual_tokens = vid_embed_dims[0] * vid_embed_dims[1]

        print("Dropping CLS token is set to:", self.drop_cls_token)
        video_encoder_output_dim = self.video_backbone.get_spatio_temporal_embed_dims(
            drop_cls=self.drop_cls_token
        )[2]
        text_decoder_input_dim = self.text_decoder.embed_dim
        if self.head_type is None or self.head_type.lower() == "none":
            self.head = nn.Identity()
        elif self.head_type.lower() == "linear":
            self.head = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(video_encoder_output_dim, text_decoder_input_dim),
            )
        elif self.head_type.lower() == "mlp":
            self.head = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(video_encoder_output_dim, video_encoder_output_dim),
                nn.GELU(),
                nn.Linear(video_encoder_output_dim, text_decoder_input_dim),
            )
        else:
            raise NotImplementedError(f"Head type {self.head_type} not implemented.")

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self.num_frozen_epochs > 0:
            if not self.backbone_is_frozen:
                self.freeze_backbone()
            if not self.text_decoder_is_frozen:
                self.freeze_text_decoder()
        elif self.current_epoch == self.num_frozen_epochs:
            if self.backbone_is_frozen:
                self.unfreeze_backbone()
            if self.text_decoder_is_frozen:
                self.unfreeze_text_decoder()
        return

    def freeze_backbone(self):
        print("Freezing backbone")
        for param in self.video_backbone.parameters():
            param.requires_grad = False
        self.video_backbone.eval()
        self.backbone_is_frozen = True
        return

    def freeze_text_decoder(self):
        print("Freezing text decoder")
        for param in self.text_decoder.parameters():
            param.requires_grad = False
        self.text_decoder.eval()
        self.text_decoder_is_frozen = True
        return

    def unfreeze_backbone(self):
        print("Unfreezing backbone")
        for param in self.video_backbone.parameters():
            param.requires_grad = True
        self.video_backbone.train()
        self.backbone_is_frozen = False
        return

    def unfreeze_text_decoder(self):
        print("Unfreezing text decoder")
        for param in self.text_decoder.parameters():
            param.requires_grad = True
        self.text_decoder.train()
        self.text_decoder_is_frozen = False
        return

    def get_prompt(self):  # TODO: Add prompts to the config
        prompt = "A video of: "
        return prompt

    def forward(self, batch):
        video = batch["video"]
        text = batch["caption"]
        prompt = self.get_prompt()
        video_features = self.video_backbone.get_spatio_temporal_embeds(
            video, drop_cls=self.drop_cls_token
        )
        # TODO: Maybe need to reshape video_features before passing to head
        # Flatten temporal dimension (b, t, h*w, d) -> (b, t*h*w, d) (keep h*w together)
        video_features = video_features.reshape(
            video_features.shape[0], -1, video_features.shape[-1]
        )
        video_features = self.head(video_features)
        text_outputs = self.text_decoder(
            text, prompt=prompt, visual_inputs=video_features
        )
        return text_outputs

    def get_labels(self, batch):
        text = batch["caption"]
        labels = self.text_decoder.get_labels(text)
        return labels

    def training_step(self, batch, batch_idx):
        batch_size = len(batch["caption"])
        text_logits = self(batch).contiguous()
        labels = self.get_labels(batch).contiguous()

        # Flatten the logits and labels
        loss = self.train_loss(
            text_logits.view(-1, text_logits.size(-1)), labels.view(-1)
        )

        # Calculate accuracy
        preds = text_logits.argmax(-1)
        acc = self.train_acc(preds.view(-1), labels.view(-1))

        self.log("train_loss", loss, batch_size=batch_size, on_step=True, prog_bar=True)
        self.log("train_acc", acc, batch_size=batch_size, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch["caption"])
        text_logits = self(batch).contiguous()
        labels = self.get_labels(batch).contiguous()

        # Shift logits and labels for loss calculation

        # Flatten the logits and labels
        loss = self.val_loss(
            text_logits.view(-1, text_logits.size(-1)), labels.view(-1)
        )

        # Calculate accuracy
        preds = text_logits.argmax(-1)
        acc = self.val_acc(preds.view(-1), labels.view(-1))

        self.log("val_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)
        self.log("val_acc", acc, batch_size=batch_size, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        backbone_params = [p for p in self.video_backbone.parameters()]

        proj_head_params = [p for p in self.head.parameters()]

        text_decoder_params = [
            p
            for name, p in self.text_decoder.named_parameters()
            if "prompt" not in name
        ]

        prompt_tuning_params = []
        if self.text_decoder.num_learnable_prompt_tokens > 0:
            prompt_tuning_params = [
                self.text_decoder.prefix_prompt_embeds.weight,
                self.text_decoder.mid_prompt_embeds.weight,
                self.text_decoder.suffix_prompt_embeds.weight,
            ]

        optimizable_params = [
            {
                "params": backbone_params,
                "lr": self.lr * self.backbone_lr_multiplier,
                "weight_decay": self.backbone_weight_decay,
                "name": "backbone",
            },
            {
                "params": proj_head_params,
                "lr": self.lr,
                "weight_decay": self.head_weight_decay,
                "name": "proj_head",
            },
            {
                "params": text_decoder_params,
                "lr": self.lr * self.text_decoder_lr_multiplier,
                "weight_decay": self.text_decoder_weight_decay,
                "name": "text_decoder",
            },
            {
                "params": prompt_tuning_params,
                "lr": self.lr * self.prompt_lr_multiplier,
                "weight_decay": self.prompt_weight_decay,
                "name": "prompt_tuning",
            },
        ]

        optimizer = torch.optim.AdamW(optimizable_params)
        scheduler = CosineAnnealingLR(optimizer, self.total_num_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
