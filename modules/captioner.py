from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

class VLMCaptioner:
    def __init__(self):
        torch.set_num_threads(4)
        self.model_id = "vikhyatk/moondream2"
        
        # Optimized for Intel Mac with Accelerate installed
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float32, 
            revision="2024-05-20", # Using a specific older, stable revision
            device_map={"": "cpu"} 
        )
        print(f"✅ Model is running on: {self.model.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.eval()

    def generate_caption(self, image):
    
        # NEW: Resize the image to a smaller resolution (e.g., 384x384)
        # This significantly speeds up the 'encode_image' step on old CPUs.
        image = image.resize((190, 190), Image.NEAREST)
    
        print("✨ Encoding image (this should take ~60 seconds)...")
        with torch.no_grad():
            # Use inference_mode for even less overhead than no_grad
            with torch.inference_mode():
                enc_image = self.model.encode_image(image)
                print("✨ Image encoded. Generating text...")
                caption = self.model.answer_question(
                    enc_image, 
                    "Describe this scene in one sentence.", 
                    self.tokenizer
                )
        return caption.strip()   