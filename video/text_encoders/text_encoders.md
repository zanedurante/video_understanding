### Text encoder API
Each text encoder has the following functionality:
- `self.get_text_embeds(texts)` that returns a tensor of shape `(b, d)`, where `b` is the batch dim and `d` is the text embedding dim. 
- `self.get_text_embed_dim()` returns an int `d` that is the text embedding dim.

All backbones should be loaded with `video.text_encoders.utils.get_textencoder(model_name, **kwargs)`.

#### Text encoder args
- `pretrained_path=None` Path or URL to the pre-trained text encoder.
- `frozen=False` Whether or not to freeze the video encoder.
- `lora=False` Whether or not to fine-tune with LoRA.  `lora` and `frozen` are mutually exclusive.
- `device='current'` Device to load the model on. if 'current' uses torch.cuda.current_device()

### Text encoders to implement
- [ ] CLIP BERT
- [ ] Bio-medical version of BERT pre-trained on healthcare data.

### TODOs:
- [ ] Allow for training of the text encoders on text-only data (.csv) with the caption field?  Can use for Q+A also maybe.