### Unified interface for text decoders

Functions all decoders should share:
- `self.get_input_dim()` returns the input dim of the text decoder (embedding dimension of the transformer).  We project into this vector dimension
- `self.forward(prompt, texts, visual_embeds=None)`