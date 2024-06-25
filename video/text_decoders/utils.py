from video.text_decoders.opt import load_opt_decoder

"""
Utils for loading text decoders (LLMs).
"""


def get_text_decoder(text_decoder_name, **kwargs):
    if text_decoder_name is None or text_decoder_name.lower() == "none":
        raise NotImplementedError("No text decoder specified in config.")
    model_type = text_decoder_name.split("_")[0]
    model_name = "_".join(text_decoder_name.split("_")[1:])

    if model_type == "opt":
        return load_opt_decoder(model_name, **kwargs)
    elif model_type == "avl":
        if model_name == "125m":
            return load_opt_decoder("125m", **kwargs)
        else: # use pretraining
            return load_opt_decoder("avl", **kwargs)

        return load_opt_decoder("125m", **kwargs)

    else:
        raise NotImplementedError("LLM {} not implemented".format(model_type))
