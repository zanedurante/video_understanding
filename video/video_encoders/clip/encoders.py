from video.video_encoders.clip.clip import load as load_clip

if __name__ == "__main__":
    model, preprocess = load_clip("ViT-B/32")
    #print(model)
    #print(model.visual)
    #print(preprocess)
    print("Done")