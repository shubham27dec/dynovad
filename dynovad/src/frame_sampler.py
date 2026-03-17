import os
import cv2
from tqdm import tqdm


def sample_frames_from_video(video_path: str, output_dir: str, frame_stride: int):
    """
    Sample frames from a single video.

    Args:
        video_path: path to video file
        output_dir: directory to save frames
        frame_stride: sample every N frames
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            out_path = os.path.join(output_dir, f"{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()


def process_videos(
    video_dir: str,
    frame_dir: str,
    frame_stride: int,
):
    """
    Process all videos and extract frames.

    Args:
        video_dir: directory containing videos
        frame_dir: output directory for frames
        frame_stride: sample every N frames
    """

    os.makedirs(frame_dir, exist_ok=True)

    video_files = sorted(
        f for f in os.listdir(video_dir) if f.endswith(".mp4") or f.endswith(".avi")
    )

    for fname in tqdm(video_files):

        video_path = os.path.join(video_dir, fname)
        out_folder = os.path.join(frame_dir, fname.replace(".mp4", "").replace(".avi", ""))

        # resumable
        if os.path.exists(out_folder) and len(os.listdir(out_folder)) > 0:
            continue

        sample_frames_from_video(video_path, out_folder, frame_stride)