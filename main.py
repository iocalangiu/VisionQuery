import modal
from io import BytesIO
from PIL import Image
from src.data_io import get_video_sources, get_cifar_sources, get_food_sources
from src.ingestion import get_pixels_from_source
from src.storage import VisionStorage

# --- CONFIGURATION ---
# Change this to "CIFAR", "FOOD101","LOCAL", or "S3" in future
#MODE = "LOCAL"

CIFAR_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def run_vision_query(mode: str = "CIFAR", limit: int = None):
    storage = VisionStorage()

    # 1. Look up the deployed Moondream worker
    try:
        # This matches the app name and class name in src/vlm_worker.py
        vlm = modal.Cls.from_name("vision-query-moondream", "MoondreamWorker")()
    except Exception as e:
        print(f"❌ Actual Error: {e}")
        return

    # 1. Strategy Picker (No commenting out!)
    if mode == "CIFAR":
        sources = get_cifar_sources(num=10)
    elif mode == "FOOD101":
        sources = get_food_sources(num=10)
    elif mode == "LOCAL":
        sources = get_video_sources(local_dir="videos/videos_v0")
    elif mode == "S3":
        print("❌ Not implemented yet")

    count = 0
    for source in sources:
        if limit and count >= limit:
            break

        print("🎬 Processing: {source.uri}")

        # Local Mac extraction
        frame = get_pixels_from_source(source)
        # frame = extract_random_frame(source)
        if frame is None:
            print("⚠️ Skipping {source.uri}: No frame extracted.")
            continue

        if mode == "CIFAR" and hasattr(source, "label"):
            # source.label is an integer (0-9)
            friendly_name = CIFAR_LABELS[source.label]

        if mode == "FOOD101" and hasattr(source, "label"):
            friendly_name = source.label

        # Convert numpy frame to bytes for transmission
        img = Image.fromarray(frame)
        if mode == "CIFAR" or img.width < 224:
            # Upscale tiny images so the AI can actually see them!
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")  # Moondream likes PNG/JPEG

        print("☁️ Calling Moondream2 on Modal...")
        try:
            caption, embedding = vlm.describe_image.remote(buf.getvalue())
            print(f"🤖 Moondream says: {caption}\n")
            print(f"🏷️  Actual: {friendly_name if (mode == 'CIFAR' or mode == 'FOOD101') else 'N/A'}")

            storage.save_result(str(source.uri), caption, embedding)
            print("💾 Successfully indexed in database.\n")
            count += 1
        except Exception as e:
            print(f"❌ Error during VLM inference for {source.uri}: {e}")


if __name__ == "__main__":
    run_vision_query()
