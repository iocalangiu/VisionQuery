# tests/test_ingestion.py
import numpy as np
from src.schema import MediaSource
from src.ingestion import extract_random_frame
from src.data_io import get_cifar_sources


def test_extract_random_frame_local():
    """
    Verifies that the ingestion engine can open a local video
    and return a valid numpy image array.
    """
    # 1. SETUP: Use a small sample video in your repo
    source = MediaSource(uri="samples/test_video.mp4", media_type="video",source_type="local")

    # 2. EXECUTE
    frame = extract_random_frame(source)

    # 3. ASSERT: The professional way to verify data
    assert frame is not None, "Extraction returned None"
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array"
    assert len(frame.shape) == 3, "Frame should have 3 dimensions (H, W, C)"
    assert frame.shape[2] == 3, "Frame should have 3 color channels (RGB)"


def test_extract_invalid_path():
    """Verifies the system handles missing files gracefully."""
    source = MediaSource(uri="non_existent.mp4", media_type="video",source_type="local")
    frame = extract_random_frame(source)
    assert frame is None

def test_cifar_generator_integrity():
    """Verify that get_cifar_sources produces valid MediaSource objects."""
    num_to_test = 5
    sources = get_cifar_sources(num=num_to_test)
    
    assert len(sources) == num_to_test
    for source in sources:
        # Pydantic validation: this will raise an error if the class changed
        assert isinstance(source, MediaSource)
        assert source.media_type == "image"
        assert source.source_type == "cifar"
        assert source.uri.isdigit() # CIFAR URIs are indices