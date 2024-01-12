from video.text_decoders.opt import (
    load_opt_model_tokenizer,
    load_opt_decoder,
    OPTTextDecoder,
)
from video.text_decoders.utils import get_text_decoder
from transformers import GPT2Tokenizer, OPTForCausalLM
from torch.nn import CrossEntropyLoss
import torch

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
    assert loss.item() < 0.20  # loss should be very low, this is an easy example

def test_prepare_inputs_for_opt():
    # Check that the prepare_inputs function is working correctly for simple text case
    prompt = "The person is walking and then"
    response = " the person is running.\nI'm not sure if you're being sarcastic or not, but I"
    decoder = load_opt_decoder("125m")
    func_input_embeds = decoder.prepare_inputs([response], prompt=prompt, visual_inputs=None)
    llm, tokenizer = load_opt_model_tokenizer("125m")
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_embeds = llm.model.decoder.embed_tokens(inputs.input_ids)
    # convert output to text
    length = len(tokenizer.encode(prompt))
    assert torch.allclose(func_input_embeds[0][:length], input_embeds[0][:length])


def test_opt_text_decoder_forward():
    debug = False
    decoder = load_opt_decoder("125m")
    prompt = "The first letter of the English alphabet is: A. The second letter of the English alphabet is:"
    texts = [" B. The third letter of the English alphabet is: C."]
    outputs = decoder(texts, prompt=prompt, visual_inputs=None)
    labels = decoder.get_labels(texts)
    
    # get argmax of outputs and outputs_no_mask
    outputs_idxs = torch.argmax(outputs, dim=-1)

    # print labels in text and outputs and outputs_no_mask in text
    if debug:
        print("==== LABELS\t\t:", decoder.tokenizer.decode(labels[0]))
        print("==== OUTPUTS\t\t:", decoder.tokenizer.decode(outputs_idxs[0]))

        print("Raw token idxs outputs:", outputs_idxs[0])
        print("Raw token idxs labels:", labels[0])

    # Flatten the logits and labels
    loss_fct = CrossEntropyLoss(
        ignore_index=-100
    )  # This will ignore the -100 indices in labels
    loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(labels.size(-1)))
    assert loss.item() > 0.0
    assert loss.item() < 0.20  # loss should be low, this is a very easy example


def test_opt_text_decoder_forward_hard():
    decoder = load_opt_decoder("125m")
    prompt = "Simple giraffe thermodynamics: "
    texts = ["Aqueducts hunger yellow bundle"]
    outputs = decoder(texts, prompt=prompt, visual_inputs=None)
    labels = decoder.get_labels(texts)
    
    # get argmax of outputs and outputs_no_mask
    outputs_idxs = torch.argmax(outputs, dim=-1)

    # Flatten the logits and labels
    loss_fct = CrossEntropyLoss(
        ignore_index=-100
    )  # This will ignore the -100 indices in labels
    loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
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
    test_prepare_inputs_for_opt()
    test_opt_text_decoder_forward()
    test_opt_text_decoder_forward_hard()
    # test_opt_text_decoder_visual_forward()
    print("All tests passed!")
