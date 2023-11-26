### Backbone API
Each backbone has the following functionality:
- `self.get_spatio_temporal_embeds(video_batch)` generally, these embeddings are structured to match a `(b, t, s, d)` format, representing the batch size, number of frames, and spatio-temporal dimension, and embedding dimension.
- `self.get_spatio_temporal_embed_dims()` returns a tuple of size `(t, s, d)`.
- `self.convert_spatio_temporal_embeds_to_video(spatio_temporal_embeds)` returns an embedding of size `(b, d)` representing the batch size and video-level embedding dimension.
- `self.get_video_level_embeds(video_batch)` gets an embedding of size `(b, d)`, representing the batch size and the video-level embedding dimension.  This calls `self.get_spatio_temporal_embeds` and then `self.convert_spatio_temporal_embeds_to_video`.
- `self.get_video_level_embed_dim()` returns `d`.

All backbones should be loaded with `video.video_encoders.utils.get_backgone`.  The arguments are `(model_name, **kwargs)`

For each backbone that does not inherently support this API, we add spatial/temporal pooling layers or use the embeddings from an earlier layer in the backbone as needed.  

#### Backbone args
- `pretrained_path=None` Path or URL to the pre-trained video encoder.
- `frozen=False` Whether or not to freeze the video encoder.
- `lora=False` Whether or not to fine-tune with LoRA.  `lora` and `frozen` are mutually exclusive.
- `device='current'` Device to load the model on. if 'current' uses torch.cuda.current_device()
- `num_frames=16` The number of frames to use as input to the model.

### Backbones to implement:
For each backbone, have a default set of weights to load (pretrained) model, can allow this to be configurable later if needed.  Ideally, each of these backbones can be seemlessly integrated with the other modules in this library.

#### CLIP-based backbones:
- [ ] CLIP, this uses frame-level CLIP embeddings and video-level temporal pooling like ViFi-CLIP
- [ ] QformerCLIP (Needs to be CLIP eva-g to get pretrained, can specify between which qformer ckpt)
- [ ] Trio (Frozen-in-Time, add argument for adding the Qformer on top)
- [ ] ViCLIP 
- [ ] Others?

#### MAE-based backbones:
- [ ]  VideoMAE
- [ ]  VideoMAEv2

#### Others
- [ ] InternVideo