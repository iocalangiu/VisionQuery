import modal
from io import BytesIO
from PIL import Image
from src.data_io import get_video_sources, get_cifar_sources, get_food_sources
from src.ingestion import get_pixels_from_source
from src.storage import VisionStorage
from itertools import islice


# Helper to chunk any generator
def chunk_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


# --- CONFIGURATION ---
# Change this to "CIFAR", "FOOD101","LOCAL", or "S3" in future
# MODE = "LOCAL"

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


def run_vision_query(mode: str = "CIFAR", limit: int = 20, batch_size: int = 8):
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
    for batch in chunk_iterable(sources, batch_size):
        if limit and count >= limit:
            break

        payloads = []
        metadata = []

        for source in batch:
            if limit and count >= limit:
                break
            print(f"🎬 Processing: {source.uri}")

            # Local Mac extraction
            frame = get_pixels_from_source(source)
            if frame is None:
                print("⚠️ Skipping {source.uri}: No frame extracted.")
                continue

            img = Image.fromarray(frame)
            if mode == "CIFAR" or img.width < 224:
                # Upscale tiny images so the AI can actually see them!
                img = img.resize((224, 224), Image.Resampling.LANCZOS)

            buf = BytesIO()
            img.save(buf, format="PNG")  # Moondream likes PNG/JPEG

            label = "N/A"

            if mode == "CIFAR" and hasattr(source, "label"):
                # source.label is an integer (0-9)
                label = CIFAR_LABELS[source.label]
            elif mode == "FOOD101" and hasattr(source, "label"):
                label = source.label

            count += 1
            payloads.append(buf.getvalue())
            metadata.append({"uri": str(source.uri), "label": label})
        if not payloads:
            continue

        print(f"Sending batch of {len(payloads)} to Modal...")
        results = list(vlm.describe_image.map(payloads))

        # Storage
        for i, (caption, embedding) in enumerate(results):
            if i < len(metadata):
                m = metadata[i]
                print(f"🤖 [{m['label']}] -> {caption[:50]}...")
                storage.save_result(m["uri"], caption, embedding)


if __name__ == "__main__":
    run_vision_query()
