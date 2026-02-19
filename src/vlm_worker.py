import modal
import io
import torch


app = modal.App("vision-query-moondream")

vlm_image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers==4.40.0",
    "tokenizers==0.19.1",
    "pillow",
    "timm",
    "einops",
    "sentence-transformers",
)


@app.cls(
    image=vlm_image, 
    gpu="T4",
    container_idle_timeout=300,  # Keeps GPU warm for 5 mins to handle next batch
    concurrency_limit=5,         # Limits total containers to save money
    allow_concurrent_inputs=4    # CRITICAL: Allows 1 GPU to process 4 images in parallel threads
)

class MoondreamWorker:
    @modal.enter()
    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sentence_transformers import SentenceTransformer
        import torch

        model_id = "vikhyatk/moondream2"
        revision = "2024-03-06"

        print("⏳ Loading Moondream2 and Embedding Model into GPU memory...")

        # load  vlm
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            revision=revision,
        ).to("cuda")

        # load encoder
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    @modal.method()
    def describe_image(
        self,
        image_bytes: bytes,
        prompt: str = (
            "Describe this scene strictly in the format: Subject, Action, Context. "
            "Example: 'A golden retriever, running, through a sunlit park.' "
            "Do not use full sentences or extra words."
        ),
    ):
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))

        with torch.inference_mode():
            enc_image = self.model.encode_image(img)
            answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
        embedding = self.encoder.encode(answer).tolist()
        return answer, embedding

    @modal.method()
    def embed_text(self, text: str):
        # This is exactly what search.py needs
        return self.encoder.encode(text).tolist()


# --- THE TEST BLOCK ---
@app.local_entrypoint()
def main():
    """
    This runs locally when you type 'modal deploy src/vlm_worker.py'
    """
    from src.ingestion import extract_random_frame
    from src.schema import VideoSource
    from PIL import Image

    video_path = "samples/test_video.mp4"
    print(f"🎬 Local Mac: Extracting frame from {video_path}...")

    # 1. Use your ingestion logic to get a frame
    source = VideoSource(uri=video_path, source_type="local")
    frame = extract_random_frame(source)

    if frame is None:
        print("❌ Failed to extract frame locally.")
        return

    # 2. Prepare the image for the cloud
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # 3. Trigger the Cloud GPU
    print("☁️ Sending pixels to Modal Cloud GPU...")
    worker = MoondreamWorker()
    result, embedding = worker.describe_image.remote(img_bytes)

    print("\n--- 🤖 Moondream Result ---")
    print(result)
    print(f"📏 Vector Length: {len(embedding)} (First few: {embedding[:3]}...)")
    print("---------------------------\n")
