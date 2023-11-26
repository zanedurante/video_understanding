from video.modules.dual_encoder import get_text_indices
import torch


def test_get_text_indices():
    texts = ["hello", "world", "hello", "world", "test"]
    good_indices = get_text_indices(texts)
    filtered_texts = [texts[i] for i in good_indices]
    assert len(filtered_texts) == 3
    assert filtered_texts[0] == "hello"
    assert filtered_texts[1] == "world"
    assert filtered_texts[2] == "test"

    embeds = torch.arange(5, 10)
    filtered_embeds = embeds[good_indices]
    assert filtered_embeds.shape == (3,)
    assert filtered_embeds[0] == 5
    assert filtered_embeds[1] == 6
    assert filtered_embeds[2] == 9


if __name__ == "__main__":
    test_get_text_indices()
    print("Done!")
