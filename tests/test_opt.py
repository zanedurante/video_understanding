from video.text_decoders.opt import (
    load_opt_model_tokenizer,
    load_opt_decoder,
    OPTTextDecoder,
)
from video.text_decoders.utils import get_text_decoder
from transformers import GPT2Tokenizer, OPTForCausalLM
from torch.nn import CrossEntropyLoss


# Dummy function that returns fake visual embeddings
def get_visual_inputs(text="a dog in a field"):
    # Tokenize the text and get the token embeddings
    llm, tokenizer = load_opt_model_tokenizer("125m")
    tokenized_text = tokenizer(text, padding=True, return_tensors="pt")
    input_ids = tokenized_text.input_ids
    token_embeddings = llm.model.decoder.embed_tokens(input_ids)
    return token_embeddings


def test_load_opt_model_tokenizer():
    llm, tokenizer = load_opt_model_tokenizer("125m")
    assert isinstance(llm, OPTForCausalLM)
    assert isinstance(tokenizer, GPT2Tokenizer)


def test_get_text_decoder():
    decoder = get_text_decoder("opt_125m")
    assert isinstance(decoder, OPTTextDecoder)


def test_opt_forward():
    prompt = "The first letter of the English alphabet is: A. The second letter of the English alphabet is: "
    texts = ["B. The third letter of the English alphabet is: C."]
    llm, tokenizer = load_opt_model_tokenizer("125m")

    # Encode the prompt and generate a response
    inputs = tokenizer(
        [prompt + text for text in texts], padding=True, return_tensors="pt"
    )
    labels = inputs.input_ids.clone()
    prompt_length = len(tokenizer.encode(prompt))
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
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    assert loss.item() > 0.0
    assert loss.item() < 0.10  # loss should be very low, this is an easy example


def test_opt_text_decoder_forward():
    decoder = load_opt_decoder("125m")
    prompt = "The first letter of the English alphabet is: A. The second letter of the English alphabet is: "
    texts = ["B. The third letter of the English alphabet is: C."]
    outputs = decoder(texts, prompt=prompt, visual_inputs=None)
    labels = decoder.get_labels(texts)

    # Shift logits and labels for loss calculation
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the logits and labels
    loss_fct = CrossEntropyLoss(
        ignore_index=-100
    )  # This will ignore the -100 indices in labels
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    assert loss.item() > 0.0
    assert loss.item() < 0.10  # loss should be very low, this is an easy example


def test_opt_text_decoder_forward_hard():
    decoder = load_opt_decoder("125m")
    prompt = "Simple giraffe thermodynamics: "
    texts = ["Aqueducts hunger yellow bundle"]
    outputs = decoder(texts, prompt=prompt, visual_inputs=None)
    labels = decoder.get_labels(texts)

    # Shift logits and labels for loss calculation
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the logits and labels
    loss_fct = CrossEntropyLoss(
        ignore_index=-100
    )  # This will ignore the -100 indices in labels
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    assert loss.item() > 5.0  # loss should be high, this is a hard example


""" TODO: Fix test!
def test_opt_text_decoder_visual_forward():
    decoder = load_opt_decoder("125m")
    prompt = "Repeat the following text: "
    visual_inputs = get_visual_inputs(text="a dog in a field. a dog")
    outputs = decoder.generate(text_batch=prompt, visual_inputs=visual_inputs)
    decoded_outputs = decoder.tokenizer.decode(outputs[0])
    assert "in a field" in decoded_outputs
"""

if __name__ == "__main__":
    # test_get_text_decoder()
    # test_load_opt_model_tokenizer()
    # test_opt_forward()
    # test_opt_text_decoder_forward()
    test_opt_text_decoder_forward_hard()
    # test_opt_text_decoder_visual_forward()
    print("All tests passed!")
