import json
import os
from pathlib import Path
from typing import Generator, List
from src.schema import MediaSource
from datasets import load_dataset

def get_video_sources(
    json_path: str | None = None, local_dir: str | None = None
) -> Generator[MediaSource, None, None]:
    """
    A generator that yields VideoSource objects from
    either a JSON metadata file or a local directory.
    """

    # Choice 1: Process JSON from a S3 metadata
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            # Handle both a single object or a list of objects
            items = data if isinstance(data, list) else [data]

            for item in items:
                video_url = item.get("outputVideo")
                if video_url:
                    yield MediaSource(
                        uri=video_url,
                        media_type="video",
                        source_type="remote",
                    )

    # Choice 2: Process local files
    if local_dir and os.path.exists(local_dir):
        valid_extensions = {".mp4", ".mov", ".avi", ".mkv"}
        for file_path in Path(local_dir).iterdir():
            if file_path.suffix.lower() in valid_extensions:
                yield MediaSource(
                    uri=file_path,
                    media_type="video", # <--- Added this
                    source_type="local",
                )

def get_cifar_sources(num: int = 10) -> List[MediaSource]:
    """
    Creates a list of MediaSource objects pointing to CIFAR-10 indices.
    """
    # We don't download the whole thing, just define the pointers
    return [
        MediaSource(uri=str(i), media_type="image", source_type="cifar") 
        for i in range(num)
    ]


# --- Usage Example ---
if __name__ == "__main__":
    # Test local
    print("Scanning local videos...")
    for source in get_video_sources(local_dir="./tests/sample_videos"):
        print(f"Found {source.source_type}: {source.uri}")

    # Test JSON
    # print("Parsing JSON...")
    # for source in get_video_sources(json_path="data.json"):
    #     print(f"Found {source.source_type}: {source.uri}")
