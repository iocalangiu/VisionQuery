import modal
import io

app = modal.App("vision-query-moondream")

# Using your pinned versions for stability
vlm_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "transformers==4.40.0", 
        "tokenizers==0.19.1", 
        "pillow", 
        "timm", 
        "einops"
    )
)

@app.cls(image=vlm_image, gpu="T4")
class MoondreamWorker:
    @modal.enter()
    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_id = "vikhyatk/moondream2"
        revision = "2024-03-06"
        
        print("⏳ Loading Moondream2 into GPU memory...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16,
            revision=revision
        ).to("cuda")

    @modal.method()
    def describe_image(self, image_bytes: bytes, prompt: str = "Describe this scene in one sentence."):
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Moondream specific inference style
        enc_image = self.model.encode_image(img)
        answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
        return answer

# --- THE TEST BLOCK ---
@app.local_entrypoint()
def main():
    """
    This runs on your 2016 Mac when you type 'modal run src/vlm_worker.py'
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
    result = worker.describe_image.remote(img_bytes)

    print("\n--- 🤖 Moondream Result ---")
    print(result)
    print("---------------------------\n")