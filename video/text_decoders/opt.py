# OPT model, using huggingface implementation
from transformers import OPTForCausalLM, GPT2Tokenizer
import torch.nn as nn
from video.text_decoders.base_decoder import BaseTextDecoder
from tokenizers.processors import TemplateProcessing
import torch

name2ckpt = {
    "avl": "zanedurante/avl_model.pth",
    "125m": "facebook/opt-125m",
    "350m": "facebook/opt-350m",
    "1.3b": "facebook/opt-1.3b",
    "2.7b": "facebook/opt-2.7b",
    "6.7b": "facebook/opt-6.7b",
    "13b": "facebook/opt-13b",
    "30b": "facebook/opt-30b",
    "66b": "facebook/opt-66b",
    "175b": "facebook/opt-175b",
}  # Names should be in format of number of params


def load_opt_decoder(opt_model_name, **kwargs):
    return OPTTextDecoder(opt_model_name, **kwargs)


class OPTTextDecoder(BaseTextDecoder):
    """
    Used like:
    """

    def __init__(self, opt_model_name, **kwargs):
        super().__init__(**kwargs)
        self.opt_model_name = opt_model_name.lower()
        self.llm, self.tokenizer = load_opt_model_tokenizer(opt_model_name)
        self.embed_dim = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size
        self.tokenizer_uses_end_token = False
        self.ignore_index = 1  # which index to ignore in the loss, ignore pad tokens
        self.add_eos_token = (
            False  # whether to add the eos token to the end of the text
        )
        if self.add_eos_token:
            self.added_eos_token = "</s>"  # Tokenize does not add the eos token in huggingface for some reason...
        else:
            self.added_eos_token = ""


def load_opt_model_tokenizer(opt_model_name, **kwargs):
    opt_model_name = opt_model_name.lower()
    if opt_model_name in name2ckpt:
        ckpt = name2ckpt[opt_model_name]
    else:
        raise NotImplementedError(
            "OPT model {} not implemented.  Only have models: {}".format(
                opt_model_name, name2ckpt.keys()
            )
        )
    if opt_model_name == "avl":  # TODO: Replace with huggingface load
        ckpt = torch.load(
            "/home/durante/code/video_understanding/checkpoints/avl_model.pth",
            map_location="cpu",
        )
        opt_model = OPTForCausalLM.from_pretrained("facebook/opt-125m", **kwargs)
        # from 50272 --> 51576
        num_addeded_tokens = 51576 - 50272

        opt_model.resize_token_embeddings(51576)  # TODO: fix this hard coded for now
        modified_state_dict = {}
        for k, v in ckpt["model"].items():
            new_k = k.replace("model.model.", "model.")
            new_k = new_k.replace("model.lm_head.", "lm_head.")
            modified_state_dict[new_k] = v
        unused = opt_model.load_state_dict(modified_state_dict, strict=False)
        tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
        # add new tokens to tokenizer
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"<|{i}|>" for i in range(num_addeded_tokens)
                ]
            }
        )
        return opt_model, tokenizer

    return OPTForCausalLM.from_pretrained(
        ckpt, **kwargs
    ), GPT2Tokenizer.from_pretrained(ckpt)


if __name__ == "__main__":
    from torch.nn import CrossEntropyLoss
    import torch

    prompt = "The first letter of the English alphabet is: A. The second letter of the English alphabet is:"  # do not predict tokens in this prompt
    texts = [
        " B. The third letter of the English alphabet is: C."
    ]  # predict every token in this text
    llm, tokenizer = load_opt_model_tokenizer("125m")

    # Encode the prompt and generate a response
    inputs = tokenizer(
        [prompt + text for text in texts], padding=True, return_tensors="pt"
    )
    print("Input ids: ", inputs.input_ids)

    labels = inputs.input_ids.clone()
    encoded_prompt = tokenizer.encode(prompt)
    print("Encoded prompt: ", encoded_prompt)
    prompt_length = len(encoded_prompt)

    labels[:, :prompt_length] = -100  # Ignore loss on prompt tokens

    outputs = llm(**inputs)
    logits = outputs.logits

    # Shift logits and labels for loss calculation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the logits and labels
    loss_fct = CrossEntropyLoss(
        ignore_index=-100
    )  # This will ignore the -100 indices in labels
    print(shift_logits.view(-1, shift_logits.size(-1)).shape)
    print(shift_labels.view(-1).shape)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print("Loss: ", loss.item())
    # print predicted values vs labels (use prompt_length - 1 since shifted)
    print(
        "Predicted: ",
        torch.argmax(
            shift_logits.view(-1, shift_logits.size(-1))[prompt_length - 1 :], dim=1
        ),
    )
    print("Labels: ", shift_labels.view(-1)[prompt_length - 1 :])
    # print in text format
    print(
        "Predicted: ",
        tokenizer.decode(
            torch.argmax(
                shift_logits.view(-1, shift_logits.size(-1))[prompt_length - 1 :], dim=1
            )
        ),
    )
    print("Labels: ", tokenizer.decode(shift_labels.view(-1)[prompt_length - 1 :]))

    # outputs = llm.generate(**inputs, max_length=77)

    # Decode the generated text
    # decoded_output = tokenizer.decode(outputs[0])

    # print(decoded_output)

    # forward_outputs = llm(**inputs)

    # print(forward_outputs.shape)
