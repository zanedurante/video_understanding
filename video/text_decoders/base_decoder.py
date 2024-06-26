import torch
import torch.nn as nn


class BaseTextDecoder(nn.Module):
    """
    Used like:
    """

    def __init__(
        self,
        text_first=True,
        num_learnable_prompt_tokens=0,
        use_start_token_for_caption=False,
        max_caption_length=128,
        **kwargs
    ):  # Subclasses need to set self.llm and self.tokenizer
        super().__init__()
        # Set these in the subclass:
        self.embed_dim = 768  # model specific, should set in constructor
        self.vocab_size = None  # model specific, should set in constructor
        self.tokenizer_uses_end_token = False  # model specific, whether the tokenizer uses an end token by default when tokenizing
        self.ignore_index = 1  # tokenizer specific, tokenizer pad token index
        self.added_eos_token = ""  # Needed for OPT, since the tokenizer does not add the eos token for some reason...
        self.max_caption_length = max_caption_length
        print("==== MAX INPUT LENGTH:", max_caption_length)

        # Hyperparameters to be set in the config:
        self.text_first = text_first  # Otherwise it uses the visual tokens first
        self.num_learnable_prompt_tokens = num_learnable_prompt_tokens  # If > 0, we learn a prompt token to place between the text, visual, and extra tokens
        self.use_start_token_for_caption = use_start_token_for_caption  # Hyperparameter, need to see whether this works better with or without this set

        # TODO: Make it more customizable how the prompt tuning is done
        # For now, learns 3 prompt tokens, one is prefix, one is between text and visual, and one is after visual and gt caption
        if (
            self.num_learnable_prompt_tokens > 0
        ):  # Prompt tokens go between the prompt text and visual tokens
            self.prefix_prompt_embeds = nn.Embedding(
                self.num_learnable_prompt_tokens, self.embed_dim
            )
            nn.init.normal_(self.prefix_prompt_embeds.weight, mean=0.0, std=0.02)
            self.mid_prompt_embeds = nn.Embedding(
                self.num_learnable_prompt_tokens, self.embed_dim
            )
            nn.init.normal_(self.mid_prompt_embeds.weight, mean=0.0, std=0.02)
            self.suffix_prompt_embeds = nn.Embedding(
                self.num_learnable_prompt_tokens, self.embed_dim
            )
            nn.init.normal_(self.suffix_prompt_embeds.weight, mean=0.0, std=0.02)

    def forward(self, text_batch, prompt=None, visual_inputs=None, **kwargs):
        # assumes text_batch is a list of strings
        inputs_embeds = self.prepare_inputs(
            text_batch, prompt=prompt, visual_inputs=visual_inputs, **kwargs
        )
        total_num_skip = 0
        if type(prompt) == list:
            total_num_skip += max([len(self.tokenizer.encode(p)) for p in prompt])
            # TODO: Ideally we can use variable length prompts without having this kind of implementation since this uses padding during forward pass...
        elif type(prompt) == str:
            # print("==== PROMPT:", prompt, "encoded as tokens:", self.tokenizer.encode(prompt))
            total_num_skip += len(self.tokenizer.encode(prompt))
        else:
            raise ValueError("prompt input needs to be list or string!")
        total_num_skip += (
            self.num_learnable_prompt_tokens * 3
        )  # 3 for prefix, mid, suffix prompt tokens
        if visual_inputs is not None:
            total_num_skip += visual_inputs.shape[1]
        if (
            self.use_start_token_for_caption
        ):  # Skip start token at the beginning of the caption
            total_num_skip += 1

        if total_num_skip > 0:
            total_num_skip -= 1  # skip the first token, since it is the start token

        # currently not using an attention mask!
        # attn_mask = self.get_custom_mask(inputs_embeds.shape[0], inputs_embeds.shape[1], total_num_skip=total_num_skip)
        # attn_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1]).int().to(self.llm.device)
        output_preds = self.llm(inputs_embeds=inputs_embeds, **kwargs)

        return output_preds.logits[
            :, total_num_skip:-1, :
        ]  # skip last token since it is predicted after the end token

    def get_labels(self, text_batch):
        text_batch = [text + self.added_eos_token for text in text_batch]
        tokenized_text = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_caption_length,
        )
        labels = tokenized_text.input_ids.clone()[:, 1:].to(
            self.llm.device
        )  # ignore start token
        return labels

    def get_custom_mask(self, batch_size, seq_length, total_num_skip=0):
        # Creates a custom mask that allows for attention between all of the first total_num_skip tokens and then is causal afterwards
        mask = torch.ones(batch_size, seq_length).int().to(self.llm.device)
        mask[:, :total_num_skip] = 0
        return mask

    def prepare_inputs(
        self,
        text_batch,
        prompt="",
        visual_inputs=None,
        extra_embeds=None,
        drop_text=False,
        **kwargs
    ):
        # Goal: predict the next token in the text_batch, given the prompt and visual inputs

        if extra_embeds is not None:
            raise NotImplementedError("Extra tokens not implemented yet")
        batch_size = len(text_batch)
        if type(prompt) == str:
            text_prompts = [prompt for _ in text_batch]  # Assumes prompt is a string
        elif type(prompt) == list:
            text_prompts = prompt
        else:
            raise ValueError("prompt input needs to be list or string!")
        text_batch = [
            text + self.added_eos_token for text in text_batch
        ]  # add eos token to batch (added_eos_token is "" for most models)
        tokenized_prompts = self.tokenizer(
            text_prompts, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to(
            self.llm.model.device
        )  # TODO: Change to remove the padding later, currently feeds padding to the model, maybe not ideal
        tokenized_text = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_caption_length,
        ).input_ids.to(self.llm.model.device)
        text_inputs = self.llm.model.decoder.embed_tokens(tokenized_prompts)
        text_targets = self.llm.model.decoder.embed_tokens(tokenized_text)[
            :, 1:, :
        ]  # Use optional start token instead
        if drop_text:  # only used during evaluation, don't give gt text
            text_targets = torch.zeros(
                text_targets.shape[0],
                text_targets.shape[1],
                self.embed_dim,
                device=text_targets.device,
            )
        start_token = torch.zeros(
            text_inputs.shape[0], 0, self.embed_dim, device=text_inputs.device
        )
        end_token = torch.zeros(
            text_inputs.shape[0], 0, self.embed_dim, device=text_inputs.device
        )

        start_token = text_inputs[:, 0, :].unsqueeze(1)  # (b, 1, d)
        text_inputs = text_inputs[:, 1:, :]

        if self.tokenizer_uses_end_token:
            end_token = text_inputs[:, -1, :].unsqueeze(1)  # (b, 1, d)
            text_inputs = text_inputs[:, :-1, :]

        if visual_inputs is None:
            visual_inputs = torch.zeros(
                text_inputs.shape[0], 0, self.embed_dim, device=text_inputs.device
            )

        optional_start_token_for_caption = torch.zeros(
            text_inputs.shape[0], 0, self.embed_dim, device=text_inputs.device
        )
        if self.use_start_token_for_caption:
            optional_start_token_for_caption = start_token.clone()

        if self.text_first:
            inputs = [
                start_token,
                text_inputs,
                visual_inputs,
                optional_start_token_for_caption,
                text_targets,
                end_token,
            ]
        else:
            inputs = [
                start_token,
                visual_inputs,
                text_inputs,
                optional_start_token_for_caption,
                text_targets,
                end_token,
            ]

        if self.num_learnable_prompt_tokens > 0:
            combined_inputs = torch.cat(
                [
                    inputs[0],
                    self.prefix_prompt_embeds.weight.unsqueeze(0).repeat(
                        batch_size, 1, 1
                    ),
                    inputs[1],
                    self.mid_prompt_embeds.weight.unsqueeze(0).repeat(batch_size, 1, 1),
                    inputs[2],
                    self.suffix_prompt_embeds.weight.unsqueeze(0).repeat(
                        batch_size, 1, 1
                    ),
                    inputs[3],
                    inputs[4],
                    inputs[5],
                ],
                dim=1,
            )
        else:
            combined_inputs = torch.cat(
                [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]],
                dim=1,
            )

        return combined_inputs

    def generate(
        self, text_batch, prompt=None, visual_inputs=None, temperature=0.0, **kwargs
    ):
        inputs_embeds = self.prepare_inputs(
            text_batch,
            prompt=prompt,
            visual_inputs=visual_inputs,
            drop_text=True,
            **kwargs
        )
        total_num_skip = 0
        if type(prompt) == list:
            total_num_skip += max([len(self.tokenizer.encode(p)) for p in prompt])
            # TODO: Ideally we can use variable length prompts without having this kind of implementation since this uses padding during forward pass...
        elif type(prompt) == str:
            # print("==== PROMPT:", prompt, "encoded as tokens:", self.tokenizer.encode(prompt))
            total_num_skip += len(self.tokenizer.encode(prompt))
        else:
            raise ValueError("prompt input needs to be list or string!")
        total_num_skip += (
            self.num_learnable_prompt_tokens * 3
        )  # 3 for prefix, mid, suffix prompt tokens
        if visual_inputs is not None:
            total_num_skip += visual_inputs.shape[1]
        if (
            self.use_start_token_for_caption
        ):  # Skip start token at the beginning of the caption
            total_num_skip += 1

        if total_num_skip > 0:
            total_num_skip -= 1  # skip the first token, since it is the start token

        # TODO: Maybe use total_num_skip?
        return self.llm.generate(inputs_embeds=inputs_embeds, **kwargs)
