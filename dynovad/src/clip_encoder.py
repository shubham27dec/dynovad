import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip


def load_clip_model(device: str):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def encode_frames_in_folder(
    frame_folder: str,
    model,
    preprocess,
    device: str,
):
    """
    Encode all frames in a folder into CLIP embeddings.
    """

    frame_files = sorted(f for f in os.listdir(frame_folder) if f.endswith(".jpg"))

    embeddings = []

    with torch.no_grad():

        for fname in frame_files:

            img_path = os.path.join(frame_folder, fname)

            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

            features = model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)

            embeddings.append(features.cpu().numpy()[0])

    return np.array(embeddings, dtype=np.float32)


def process_clip_embeddings(
    frame_dir: str,
    embedding_dir: str,
    device: str = "cuda",
):
    """
    Process all frame folders and save CLIP embeddings.
    """

    os.makedirs(embedding_dir, exist_ok=True)

    model, preprocess = load_clip_model(device)

    video_folders = sorted(os.listdir(frame_dir))

    for folder in tqdm(video_folders):

        in_path = os.path.join(frame_dir, folder)
        out_path = os.path.join(embedding_dir, f"{folder}.npy")

        # resumable
        if os.path.exists(out_path):
            continue

        embeddings = encode_frames_in_folder(
            in_path,
            model,
            preprocess,
            device,
        )

        np.save(out_path, embeddings)