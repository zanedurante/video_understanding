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
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

nltk.download("punkt")

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
        self.debug = False  # set to true for debugging prints
        backbone_name = get_val_from_config(config, "model.backbone_name")
        backbone_pretrained_ckpt = get_val_from_config(
            config, "model.backbone_pretrained_ckpt", None
        )
        self.num_frames = get_val_from_config(config, "data.num_frames")
        text_decoder_name = get_val_from_config(config, "model.text_decoder_name")
        self.max_caption_length = get_val_from_config(
            config, "model.max_input_length", 70
        )
        self.video_backbone = get_backbone(
            backbone_name,
            num_frames=self.num_frames,
            pretrained_path=backbone_pretrained_ckpt,
        ).float()  # TODO: How to make precision configurable
        if backbone_pretrained_ckpt is not None:
            self.video_backbone._load_pretrained()
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
            max_caption_length=self.max_caption_length,
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
        self.prompt = get_val_from_config(config, "model.prompt", "A video of: ")
        self.lr = get_val_from_config(config, "trainer.lr", 1e-4)
        self.batch_size = get_val_from_config(config, "trainer.batch_size", 16)
        self.val_batch_size = get_val_from_config(
            config, "trainer.val_batch_size", self.batch_size
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.text_decoder.vocab_size
        )
        ignore_index = self.text_decoder.ignore_index
        self.train_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )  # Pad token is 1
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.text_decoder.vocab_size
        )
        self.val_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.head_weight_decay = get_val_from_config(
            config, "trainer.head_weight_decay", 0.0
        )
        self.backbone_weight_decay = get_val_from_config(
            config, "model.backbone_weight_decay", 0.0
        )
        self.text_decoder_weight_decay = get_val_from_config(
            config, "model.text_decoder_weight_decay", 0.0
        )
        self.prompt_weight_decay = get_val_from_config(
            config, "model.prompt_weight_decay", 0.0
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
            # TODO: Fix this (hacky rn), load from AVL model
            # if "avl_pretrain" in text_decoder_name:
            #    print("Manually loading linear layer from the AVL ckpt!")
            #    ckpt = torch.load("/home/durante/code/video_understanding/checkpoints/avl_model.pth", map_location="cpu")
            #
            #    self.head[1].weight.data = ckpt["model"]["linear_projection.weight"]
            #    self.head[1].bias.data = ckpt["model"]["linear_projection.bias"]
            #    del ckpt
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
        return self.prompt

    def forward(self, batch):
        video = batch["video"]
        prompt = self.get_prompt()
        if prompt == "use_question":
            prompt = batch["question"]
            text = batch["answer"]
        else:
            text = batch["caption"]
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

    def generate(self, batch, temperature=0.0):
        video = batch["video"]
        prompt = self.get_prompt()
        if prompt == "use_question":
            prompt = batch["question"]
            text = batch["answer"]
        else:
            text = batch["caption"]
        video_features = self.video_backbone.get_spatio_temporal_embeds(
            video, drop_cls=self.drop_cls_token
        )
        # TODO: Maybe need to reshape video_features before passing to head
        # Flatten temporal dimension (b, t, h*w, d) -> (b, t*h*w, d) (keep h*w together)
        video_features = video_features.reshape(
            video_features.shape[0], -1, video_features.shape[-1]
        )
        video_features = self.head(video_features)
        text_outputs = self.text_decoder.generate(
            text,
            prompt=prompt,
            visual_inputs=video_features,
            temperature=temperature,
        )
        return text_outputs

    def get_labels(self, batch):
        prompt = self.get_prompt()
        if prompt == "use_question":
            text = batch["answer"]
        else:
            text = batch["caption"]
        labels = self.text_decoder.get_labels(text)
        return labels

    def training_step(self, batch, batch_idx):
        batch_size = len(batch["video"])
        text_logits = self(batch).contiguous()
        labels = self.get_labels(batch).contiguous()

        if self.debug:
            print("Labels: ", self.text_decoder.tokenizer.decode(labels[0]))

        # Flatten the logits and labels
        loss = self.train_loss(
            text_logits.view(-1, text_logits.size(-1)), labels.view(-1)
        )

        # Calculate accuracy
        preds = text_logits.argmax(-1)

        if self.debug:
            print("Preds:  ", self.text_decoder.tokenizer.decode(preds[0]))

        acc = self.train_acc(preds.view(-1), labels.view(-1))

        self.log("train_loss", loss, batch_size=batch_size, on_step=True, prog_bar=True)
        self.log(
            "train_acc",
            acc,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch["video"])
        if batch_size > 1:
            raise ValueError(
                "Validation batch size must be 1 for current metric calculations (BLEU-4)"
            )
        text_logits = self(batch).contiguous()
        labels = self.get_labels(batch).contiguous()

        if self.debug:
            print("\nLabels: ", self.text_decoder.tokenizer.decode(labels[0]))

        # Flatten the logits and labels
        loss = self.val_loss(
            text_logits.view(-1, text_logits.size(-1)), labels.view(-1)
        )

        perplexity = np.exp(loss.item())

        # BLEU-4 code starts here
        generated_tokens = self.generate(batch)
        text = self.text_decoder.tokenizer.decode(generated_tokens[0])
        if self.debug:
            print("generated text:", text)
        # TODO: Add BLEU-4 metric here
        # gt_text = ""
        # if "caption" in batch:
        #    gt_text = batch["caption"][0]
        # elif "answer" in batch:
        #    gt_text = batch["answer"][0]
        # else:
        #    raise ValueError("No ground truth text found in batch when computing BLEU-4 score")
        # assert type(gt_text) == str
        # assert type(text) == str
        # reference_tokens = [word_tokenize(gt_text)]
        # candidate_tokens = word_tokenize(text)
        # score = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        # self.log("bleu-4", score, on_step=False, on_epoch=True, sync_dist=True)
        # BLEU-4 code ends here

        # Calculate accuracy
        preds = text_logits.argmax(-1)

        if self.debug:
            print("Preds:  ", self.text_decoder.tokenizer.decode(preds[0]))

        acc = self.val_acc(preds.view(-1), labels.view(-1))
        if self.debug:
            print("Batch acc: ", acc)

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
            acc,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_perplexity",
            perplexity,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

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
