from video.video_encoders.clip import load_clip_backbone
from video.text_encoders.utils import get_text_encoder
from video.video_encoders.clip.clip import load as load_clip
from video.video_encoders.clip.clip import tokenize
from PIL import Image
from video.utils.video_reader import load_video
from video.datasets.dataset import get_transforms
import torch


def test_load_clip_backbone():
    model = load_clip_backbone("ViT-B/32")
    assert model is not None
    assert hasattr(model, "logit_scale")


# use cat and dog test images
def test_clip_simple_example():
    model, preprocess = load_clip("ViT-L/14@336px")
    image0 = preprocess(Image.open("tests/test_imgs/cat0.jpg")).unsqueeze(0).cuda()
    image1 = preprocess(Image.open("tests/test_imgs/cat1.jpg")).unsqueeze(0).cuda()
    image2 = preprocess(Image.open("tests/test_imgs/cat2.jpg")).unsqueeze(0).cuda()
    image3 = preprocess(Image.open("tests/test_imgs/cat3.jpg")).unsqueeze(0).cuda()
    text = tokenize(["a photo of a cat", "a photo of a dog"]).cuda()

    images = torch.cat([image0, image1, image2, image3], dim=0)

    # encode image and text
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

    # cosine similarity as logits
    logits_per_image = image_features @ text_features.t()
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # should be cat with high probability for all images
    assert (
        probs[0][0] > 0.99
    ), f"Probability of cat for cat image was only: {probs[0][0]}"
    assert (
        probs[1][0] > 0.99
    ), f"Probability of cat for cat image was only: {probs[1][0]}"
    assert (
        probs[2][0] > 0.99
    ), f"Probability of cat for cat image was only: {probs[2][0]}"
    assert (
        probs[3][0] > 0.99
    ), f"Probability of cat for cat image was only: {probs[3][0]}"


def test_clip_simple_video_example():
    model = load_clip_backbone("ViT-B/16", num_frames=4)
    clip_model, preprocess = load_clip("ViT-B/16")

    video, loaded_correctly = load_video("tests/test_imgs/cat.mp4", num_frames=4)
    text = tokenize(["a photo of a cat", "a photo of a dog"]).cuda()

    # make sure video is loaded correctly
    assert loaded_correctly, "Video tests/test_imgs/cat.mp4 was not loaded correctly"

    # preprocess video
    transforms = get_transforms("test")

    video = transforms(video)

    # encode video and text
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            video_features = model.get_video_level_embeds(video.unsqueeze(0).cuda())
            text_features = clip_model.encode_text(text)

    # cosine similarity as logits
    logits_per_video = video_features @ text_features.t()

    probs = logits_per_video.softmax(dim=-1).cpu().numpy()

    # should be cat with high probability for video
    assert (
        probs[0][0] > 0.99
    ), f"Probability of cat for cat video was only: {probs[0][0]}"


def text_load_clip_text_encoder():
    model = get_text_encoder("clip_ViT-B/32")
    assert model is not None


def test_text_embeds():
    model = get_text_encoder("clip_ViT-B/32")
    with torch.no_grad():
        text_embeds = model.get_text_embeds(
            [
                "a photo of a cat",
                "a photo of a cat",
                "a photo of two kittens",
                "a photo of a dog",
                "a photo of a bike",
            ]
        )
    embed_dim = model.get_text_embed_dim()
    assert text_embeds.shape == (5, embed_dim)

    # normalize
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # cosine similarity between embeds
    similarity = text_embeds @ text_embeds.t()

    # get the first row (first text embed) and make sure similarities are correct
    first_row = similarity[0]
    assert first_row[0] == first_row[1]
    assert first_row[1] > first_row[2]
    assert first_row[2] > first_row[4]
    assert first_row[3] > first_row[4]


if __name__ == "__main__":
    test_load_clip_backbone()  # test_clip_simple_video_example()
    print("All tests passed!")
