import modal
from io import BytesIO
from PIL import Image
from src.data_io import get_video_sources
from src.ingestion import extract_random_frame
from src.storage import VisionStorage


def run_vision_query():
    storage = VisionStorage()

    # 1. Look up the deployed Moondream worker
    try:
        # This matches the app name and class name in src/vlm_worker.py
        vlm = modal.Cls.from_name("vision-query-moondream", "MoondreamWorker")()
    except Exception as e:
        print(f"❌ Actual Error: {e}")
        return

    # 2. Iterate through your samples folder
    sources = get_video_sources(local_dir="videos/videos_v0")

    for source in sources:
        print(f"🎬 Processing: {source.uri}")

        # Local Mac extraction
        frame = extract_random_frame(source)
        if frame is None:
            print(f"⚠️ Skipping {source.uri}: No frame extracted.")
            continue

        # Convert numpy frame to bytes for transmission
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format="PNG")  # Moondream likes PNG/JPEG

        print("☁️ Calling Moondream2 on Modal...")
        try:
            caption, embedding = vlm.describe_image.remote(buf.getvalue())
            print(f"🤖 Moondream says: {caption}\n")

            storage.save_result(str(source.uri), caption, embedding)
            print("💾 Successfully indexed in database.\n")
        except Exception as e:
            print(f"❌ Error during VLM inference for {source.uri}: {e}")


if __name__ == "__main__":
    run_vision_query()
