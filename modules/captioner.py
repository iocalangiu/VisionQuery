from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

class VLMCaptioner:
    def __init__(self):
        self.model_id = "vikhyatk/moondream2"
        # We remove 'device_map' to avoid the ImportError
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float32 
        ).to("cpu") # Manually move to CPU after loading
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def generate_caption(self, image_path):
        image = Image.open(image_path)
        enc_image = self.model.encode_image(image)
        return self.model.answer_question(enc_image, "Describe this video scene.", self.tokenizer)