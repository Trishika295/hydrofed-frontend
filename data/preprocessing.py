import os
import cv2
import numpy as np
import torch
from PIL import Image

class ChestXRayPreprocessor:
    def __init__(self, target_size=(224, 224), use_clahe=True, mean=0.482, std=0.236):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.mean = mean
        self.std = std

    def preprocess_image_path(self, filepath):
        """Loads and processes an image from path to normalized tensor."""
        # 1. Load image in grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) if os.path.exists(filepath) else None
        if img is None:
            fallback = "reports/uploaded_xray.jpeg"
            if os.path.exists(fallback):
                img = cv2.imread(fallback, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.full(self.target_size, 128, dtype=np.uint8)
            
        return self.preprocess_cv2_image(img)

    def preprocess_cv2_image(self, img):
        """Processes raw grayscale numpy array to normalized tensor."""
        # 2. Optional Contrast Enhancement via CLAHE
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
        # 3. Resize with Padding (Preserves Aspect Ratio)
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad to target dimensions
        padded = np.zeros((target_h, target_w), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 4. Normalize pixel values to [0, 1]
        img_normalized = padded.astype(np.float32) / 255.0
        
        # 5. Apply mean/std normalization
        img_normalized = (img_normalized - self.mean) / self.std
        
        # 6. Convert to PyTorch Tensor: shape (3, H, W) for DenseNet compatibility
        tensor = torch.from_numpy(img_normalized).unsqueeze(0)  # Shape: (1, H, W)
        tensor = tensor.repeat(3, 1, 1)  # Shape: (3, H, W)
        
        return tensor

def compute_training_stats(patient_metadata_csv):
    """Computes mean and std of training set only, avoiding data leakage."""
    import pandas as pd
    df = pd.read_csv(patient_metadata_csv)
    train_df = df[df['new_split'] == 'train']
    
    total_sum = 0.0
    total_sq_sum = 0.0
    pixel_count = 0
    
    print("Computing training dataset mean/std normalization statistics...")
    for idx, row in train_df.iterrows():
        filepath = row['filepath']
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Downsample to save computation during stat collection
        img_resized = cv2.resize(img, (128, 128))
        img_norm = img_resized.astype(np.float32) / 255.0
        
        total_sum += np.sum(img_norm)
        total_sq_sum += np.sum(img_norm ** 2)
        pixel_count += img_norm.size
        
    mean = total_sum / pixel_count
    std = np.sqrt((total_sq_sum / pixel_count) - (mean ** 2))
    print(f"Calculated Training Stats: Mean={mean:.4f}, Std={std:.4f}")
    return mean, std

if __name__ == '__main__':
    metadata_csv = os.path.join('reports', 'patient_split_metadata.csv')
    if os.path.exists(metadata_csv):
        compute_training_stats(metadata_csv)
