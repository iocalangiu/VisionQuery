import cv2
import random
import requests
import tempfile
import numpy as np
from typing import Optional, List
from src.schema import VideoSource
import os


def download_to_temp(url: str) -> str:
    """
    Downloads a remote video to a temporary file.
    Essential for 2016 Macs where OpenCV might not support streaming URLs.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            temp_file.write(chunk)
    return temp_file.name


def extract_random_frame(source: VideoSource) -> Optional[np.ndarray]:
    """
    High-level function to extract one random RGB frame from a VideoSource.
    """
    video_path = str(source.uri)
    is_remote = source.source_type == "remote"

    # Logic for remote files
    if is_remote:
        # download locally because OpenCV seek is faster/more stable on disk
        video_path = download_to_temp(video_path)

    try:
        frame = _get_frame_at_random(video_path)
        return frame
    finally:
        # Clean up the temp file if it was remote
        if is_remote and video_path and os.path.exists(video_path):
            os.remove(video_path)


def _get_frame_at_random(path: str) -> Optional[np.ndarray]:
    """
    Internal helper that performs the OpenCV 'Seek' operation.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # Pick a random point
    random_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)

    success, frame = cap.read()
    cap.release()

    if success:
        # Convert BGR (OpenCV) to RGB (AI Models)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# --- Placeholders for future selection methods ---


def extract_kmeans_frames(source: VideoSource, k: int = 5) -> List[np.ndarray]:
    """Placeholder: Extract k representative frames using clustering."""
    print("K-Means extraction not yet implemented.")
    return []


def extract_scene_changes(source: VideoSource) -> List[np.ndarray]:
    """Placeholder: Extract frames where PySceneDetect finds a cut."""
    print("Scene detection not yet implemented.")
    return []
