import modal
from io import BytesIO
from PIL import Image
from src.data_io import get_video_sources
from src.ingestion import extract_random_frame

def run_vision_query():
    # 1. Look up the deployed Moondream worker
    try:
        # This matches the app name and class name in src/vlm_worker.py
        vlm = modal.Cls.lookup("vision-query-moondream", "MoondreamWorker")()
    except Exception as e:
        print("❌ Could not find the Cloud Worker. Run 'modal deploy src/vlm_worker.py' first.")
        return

    # 2. Iterate through your samples folder
    sources = get_video_sources(local_dir="samples")

    for source in sources:
        print(f"🎬 Processing: {source.uri}")
        
        # Local Mac extraction
        frame = extract_random_frame(source)
        if frame is None: continue

        # Convert numpy frame to bytes for transmission
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format="PNG") # Moondream likes PNG/JPEG
        
        print("☁️ Calling Moondream2 on Modal...")
        caption = vlm.describe_image.remote(buf.getvalue())
        
        print(f"🤖 Moondream says: {caption}\n")

if __name__ == "__main__":
    run_vision_query()