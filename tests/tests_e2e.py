import pytest
from unittest.mock import MagicMock, patch
import shutil
from main import run_vision_query
from pathlib import Path


@pytest.fixture
def setup_local_data():
    """Sets up a temporary video directory and cleans it up after the test."""
    test_dir = Path("videos/videos_v0")
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test.mp4").touch()

    yield test_dir  # The test runs while it's "paused" here

    # Teardown: Remove the directory and the fake file
    if test_dir.exists():
        shutil.rmtree(test_dir)


@patch("modal.Cls.from_name")
@pytest.mark.parametrize("mode", ["CIFAR", "LOCAL", "FOOD101"])
def test_pipeline_execution(mock_modal_cls, mode, setup_local_data):
    # 1. Mock the VLM Worker instance
    mock_worker_instance = MagicMock()

    # 2. Mock the .map() method specifically!
    # It needs to return a LIST of tuples because .map() handles batches.
    mock_worker_instance.describe_image.map.return_value = [
        ("A test image", [0.1] * 384)
    ]

    # 3. Handle the Modal Cls lookup
    # modal.Cls.from_name(...)() returns the instance
    mock_modal_cls.return_value = lambda: mock_worker_instance

    try:
        # Pass a small limit to keep the test fast
        run_vision_query(mode=mode, limit=1, batch_size=1)
    except Exception as e:
        pytest.fail(f"Pipeline crashed in {mode} mode with error: {e}")
