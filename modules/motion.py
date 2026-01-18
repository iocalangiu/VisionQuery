import torch
import cv2
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class MotionAnalyzer:
    def __init__(self):
        # r2plus1d_18 is the lightweight version (works for CPU)
        self.weights = R2Plus1D_18_Weights.DEFAULT
        self.model = r2plus1d_18(weights=self.weights).eval()
        self.transform = self.weights.transforms()

    def get_motion_features(self, video_path):
        # can sample 16 frames here
        # For now, return a placeholder representing the R(2+1)D output
        return {"motion_type": "high_activity", "confidence": 0.92}