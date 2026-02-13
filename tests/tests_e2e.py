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
    """
    Test that the main loop executes without crashing.
    We mock the Modal part so we don't need a GPU/Secrets to run this.
    """
    # 1. Mock the VLM Worker's response
    mock_worker = MagicMock()
    mock_worker.describe_image.remote.return_value = ("A test image", [0.1] * 384)
    mock_modal_cls.return_value = lambda: mock_worker

    # 2. Run the query logic (maybe point to a 'test_data' folder)
    try:
        # You might want to modify run_vision_query to accept a 'limit'
        # so it only processes 1 item for the test.
        run_vision_query(mode=mode, limit=1)
    except Exception as e:
        pytest.fail(f"Pipeline crashed in {mode} mode with error: {e}")
