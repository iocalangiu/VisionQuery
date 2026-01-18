from motion import MotionAnalyzer
from captioner import VLMCaptioner
import datetime
import json

# This script coordinates the work and prepares the JSON for Elasticsearch index.

def process_video(path):
    # 1. Initialize Modules
    motion_unit = MotionAnalyzer()
    vlm_unit = VLMCaptioner()

    # 2. Extract Features
    print("Analyzing motion...")
    motion_data = motion_unit.get_motion_features(path)

    print("Generating caption...")
    # (Pretend we extracted a frame at 00:02 seconds)
    caption = vlm_unit.generate_caption("frame_0002.jpg") 

    # 3. Create JSON Log
    log_entry = {
        "video_id": path.split("/")[-1],
        "timestamp": datetime.datetime.now().isoformat(),
        "vlm_description": caption,
        "motion_analysis": motion_data,
        "model_version": "LFM-2.5-VL-1.6B"
    }

    return log_entry

if __name__ == "__main__":
    result = process_video("my_video.mp4")
    print(json.dumps(result, indent=2))