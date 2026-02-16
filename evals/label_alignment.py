import sys
from pathlib import Path
import numpy as np
import gensim.downloader as api
from scipy.spatial.distance import cosine

import modal
from io import BytesIO
from PIL import Image, ImageFile
# Assuming AlignmentEvaluator, get_food_sources, and get_pixels_from_source are imported

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from main import run_vision_query
from src.data_io import get_food_sources
from src.ingestion import get_pixels_from_source

class AlignmentEvaluator:
    def __init__(self, method="word2vec"):
        self.method = method
        if method == "word2vec":
            print("⏳ Loading Word2Vec (glove-wiki-gigaword-100)...")
            self.model = api.load("glove-wiki-gigaword-100")
        elif method == "transformer":
            # Placeholder for future transformer implementation
            raise NotImplementedError("Transformer strategy coming soon!")

    def _get_word_sim(self, target_word, caption_words):
        """Finds max similarity between one target word and all caption words."""
        if target_word not in self.model:
            return 0.0
        
        target_vec = self.model[target_word]
        similarities = [
            1 - cosine(target_vec, self.model[w]) 
            for w in caption_words if w in self.model
        ]
        return max(similarities) if similarities else 0.0

    def compute_score(self, ground_truth, caption):
        """Computes semantic alignment score."""
        caption_words = caption.lower().replace(",", "").split()
        gt_parts = ground_truth.lower().split() # Handles "apple pie" -> ["apple", "pie"]

        # Calculate max similarity for each part of the GT label
        part_scores = [self._get_word_sim(part, caption_words) for part in gt_parts]
        
        # Return average score of all parts
        return np.mean(part_scores) if part_scores else 0.0

def run_alignment_report(n=10):
    evaluator = AlignmentEvaluator(method="word2vec")
    sources = get_food_sources(num=n)
    
    # 1. Look up the worker ONCE outside the loop for speed
    try:
        Moondream = modal.Cls.from_name("vision-query-moondream", "MoondreamWorker")
        vlm = Moondream()
    except Exception as e:
        print(f"❌ Error connecting to Modal: {e}")
        return

    print(f"\n🍱 Running REAL Alignment Eval on {n} images...")
    print(f"{'ID':<5} | {'Ground Truth':<15} | {'Score':<10} | {'Status'}")
    print("-" * 60)

    total_score = 0
    for source in sources:
        # 2. Extract pixels
        frame = get_pixels_from_source(source)
        if frame is None:
            continue
            
        # 3. Convert to Bytes for the worker
        img = Image.fromarray(frame)
        
        # Ensure image is at least 224x224 for Moondream's vision encoder
        if img.width < 224 or img.height < 224:
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # 4. Call the worker and get the real prediction
        try:
            prediction, _ = vlm.describe_image.remote(image_bytes)
        except Exception as e:
            print(f"⚠️ Worker error for {source.uri}: {e}")
            prediction = ""

        # 5. Score it against the ground truth
        score = evaluator.compute_score(source.label, prediction)
        total_score += score
        
        status = "✅" if score > 0.6 else "❌"
        print(f"DEBUG: VLM said: \"{prediction}\"") 
        print(f"{str(source.uri)[:5]:<5} | {prediction:<15.15} | {str(source.label):<15} | {score:<10.2f} | {status}")

    print("-" * 60)
    print(f"📈 Aggregate Pipeline Alignment: {total_score/n:.2f}")

if __name__ == "__main__":
    run_alignment_report()