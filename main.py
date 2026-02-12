import modal
from io import BytesIO
from PIL import Image
from src.data_io import get_video_sources, get_cifar_sources
from src.ingestion import get_pixels_from_source
from src.storage import VisionStorage

# --- CONFIGURATION ---
# Change this to "CIFAR", "LOCAL", or "S3" in future
MODE = "LOCAL"

CIFAR_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

def run_vision_query(limit: int = None):
    storage = VisionStorage()

    # 1. Look up the deployed Moondream worker
    try:
        # This matches the app name and class name in src/vlm_worker.py
        vlm = modal.Cls.from_name("vision-query-moondream", "MoondreamWorker")()
    except Exception as e:
        print(f"❌ Actual Error: {e}")
        return

    # 1. Strategy Picker (No commenting out!)
    if MODE == "CIFAR":
        sources = get_cifar_sources(num=10)
    elif MODE == "LOCAL":
        sources = get_video_sources(local_dir="videos/videos_v0")
    elif MODE == "S3":
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


        if MODE == "CIFAR" and hasattr(source, 'label'):
            # source.label is an integer (0-9)
            friendly_name = CIFAR_LABELS[source.label]
        

        # Convert numpy frame to bytes for transmission
        img = Image.fromarray(frame)
        if MODE == "CIFAR":
            # Upscale tiny images so the AI can actually see them!
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")  # Moondream likes PNG/JPEG

        print("☁️ Calling Moondream2 on Modal...")
        try:
            caption, embedding = vlm.describe_image.remote(buf.getvalue())
            print(f"🤖 Moondream says: {caption}\n")
            print(f"🏷️  Actual: {friendly_name if MODE == 'CIFAR' else 'N/A'}")

            storage.save_result(str(source.uri), caption, embedding)
            print("💾 Successfully indexed in database.\n")
            count += 1
        except Exception as e:
            print(f"❌ Error during VLM inference for {source.uri}: {e}")


if __name__ == "__main__":
    run_vision_query()
