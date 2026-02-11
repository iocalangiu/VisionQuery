import pytest
from unittest.mock import MagicMock, patch
from main import run_vision_query


@patch("modal.Cls.from_name")
def test_pipeline_execution(mock_modal_cls):
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
        run_vision_query(limit=1)
    except Exception as e:
        pytest.fail(f"Pipeline crashed with error: {e}")
