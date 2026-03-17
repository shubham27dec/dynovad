import os
import numpy as np
from tqdm import tqdm


def create_segments(
    spikes: np.ndarray,
    std_factor: float,
    min_threshold: float,
    min_length: int,
):
    mean = spikes.mean()
    std = spikes.std()

    threshold = max(mean + std_factor * std, min_threshold)

    segments = []

    start = 0
    current_beta = 0.0

    for i in range(1, len(spikes)):

        if spikes[i] > threshold and (i - start) >= min_length:

            segments.append([start, i, current_beta])

            start = i
            current_beta = spikes[i]

    segments.append([start, len(spikes), current_beta])

    return np.array(segments), threshold


def process_segments(
    spike_dir: str,
    segment_dir: str,
    std_factor: float,
    min_threshold: float,
    min_length: int,
):
    os.makedirs(segment_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(spike_dir) if f.endswith(".npy"))

    for fname in tqdm(files):

        in_path = os.path.join(spike_dir, fname)
        out_path = os.path.join(segment_dir, fname)

        # resumable
        if os.path.exists(out_path):
            continue

        spikes = np.load(in_path)

        segments, _ = create_segments(
            spikes,
            std_factor,
            min_threshold,
            min_length,
        )

        np.save(out_path, segments)