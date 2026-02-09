# tests/test_ingestion.py
import pytest
import numpy as np
from src.schema import VideoSource
from src.ingestion import extract_random_frame

def test_extract_random_frame_local():
    """
    Verifies that the ingestion engine can open a local video 
    and return a valid numpy image array.
    """
    # 1. SETUP: Use a small sample video in your repo
    source = VideoSource(
        uri="samples/test_video.mp4", 
        source_type="local"
    )
    
    # 2. EXECUTE
    frame = extract_random_frame(source)
    
    # 3. ASSERT: The professional way to verify data
    assert frame is not None, "Extraction returned None"
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array"
    assert len(frame.shape) == 3, "Frame should have 3 dimensions (H, W, C)"
    assert frame.shape[2] == 3, "Frame should have 3 color channels (RGB)"

def test_extract_invalid_path():
    """Verifies the system handles missing files gracefully."""
    source = VideoSource(uri="non_existent.mp4", source_type="local")
    frame = extract_random_frame(source)
    assert frame is None