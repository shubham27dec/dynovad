import os
import numpy as np
from tqdm import tqdm


def compute_semantic_spikes(embeddings: np.ndarray, k: int) -> np.ndarray:
    n, d = embeddings.shape
    spikes = np.zeros(n, dtype=np.float32)

    running_sum = np.zeros(d, dtype=np.float32)

    for i in range(1, n):

        running_sum += embeddings[i - 1]

        if i > k:
            running_sum -= embeddings[i - k - 1]

        window = min(i, k)

        context = running_sum / window
        context = context / (np.linalg.norm(context) + 1e-8)

        cos_sim = np.dot(embeddings[i], context)

        spikes[i] = 1 - cos_sim

    return spikes


def process_semantic_spikes(
    embedding_dir: str,
    spike_dir: str,
    context_window: int,
):
    os.makedirs(spike_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(embedding_dir) if f.endswith(".npy"))

    for fname in tqdm(files):

        in_path = os.path.join(embedding_dir, fname)
        out_path = os.path.join(spike_dir, fname)

        # resumable
        if os.path.exists(out_path):
            continue

        embeddings = np.load(in_path)

        spikes = compute_semantic_spikes(embeddings, context_window)

        np.save(out_path, spikes)