import modal
import io
from pathlib import Path

# --- BRICK 1: THE STABLE INFRASTRUCTURE ---
app = modal.App("vision-query-moondream")

# We pin 'transformers' and 'tokenizers' to prevent the Enum error
image = (
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

# --- BRICK 2: THE BRAIN (With Version Pinning) ---
@app.function(image=image, gpu="T4")
def generate_caption_cloud(image_bytes):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    import torch

    model_id = "vikhyatk/moondream2"
    # Pinning the revision to a known working date
    revision = "2024-03-06"
    
    print("⏳ Loading model into GPU memory...")
    
    # Load tokenizer and model using the specific revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        revision=revision
    ).to("cuda")
    
    # Process the image
    img = Image.open(io.BytesIO(image_bytes))
    enc_image = model.encode_image(img)
    answer = model.answer_question(enc_image, "Describe this scene in one sentence.", tokenizer)
    
    return answer

# --- BRICK 3: THE LOCAL INTERFACE ---
@app.local_entrypoint()
def main():
    image_path = Path("utils/test_frame.png")
    
    if not image_path.exists():
        print(f"❌ Error: Please put an image named '{image_path}' in this folder.")
        return

    print("🚀 Sending image to cloud GPU...")
    image_data = image_path.read_bytes()
    
    try:
        result = generate_caption_cloud.remote(image_data)
        print("\n--- 🤖 VisionQuery Result ---")
        print(result)
        print("-----------------------------\n")
    except Exception as e:
        print(f"❌ Professional Debug: The cloud ran into an issue: {e}")