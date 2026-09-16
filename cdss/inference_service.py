import time
import os
import torch
import numpy as np
from models.multimodal_model import HydroFedMultimodalModel
from models.uncertainty import estimate_mc_uncertainty
from xai.gradcam import GradCAM
from xai.gradcam_plus import GradCAMPlusPlus
from xai.visualization import save_xai_visualization, overlay_heatmap
from data.preprocessing import ChestXRayPreprocessor

class InferenceService:
    def __init__(self, model_path=None, num_classes=2, embed_dim=128, device='cpu'):
        self.device = torch.device(device)
        self.model = HydroFedMultimodalModel(embed_dim=embed_dim, num_classes=num_classes, pretrained_backbone=False)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Inference model loaded weights from: {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load weights from {model_path} ({e}). Using initialized weights.")
                
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = ChestXRayPreprocessor()

    def run_cdss_diagnostics(self, image_path, clinical_vector, output_vis_dir='reports/xai_outputs', visit_id='demo_visit'):
        """
        Runs the complete visual and clinical diagnostic pipeline for CDSS.
        clinical_vector: list or array of shape (5,) -> [age_norm, gender, diabetes, smoke, family]
        """
        os.makedirs(output_vis_dir, exist_ok=True)
        start_time = time.time()
        
        # 1. Preprocess Image
        xray_tensor = self.preprocessor.preprocess_image_path(image_path)
        xray_tensor = xray_tensor.unsqueeze(0).to(self.device)  # Add batch dim -> (1, 3, 224, 224)
        
        clinical_tensor = torch.tensor(clinical_vector, dtype=torch.float32).unsqueeze(0).to(self.device)  # shape: (1, 5)
        
        # 2. Measure core latency and run predictions
        t0 = time.time()
        with torch.no_grad():
            logits = self.model(xray_tensor, clinical_tensor)
            prob_pneu = torch.softmax(logits, dim=1)[:, 1].item()
        core_latency = time.time() - t0
        
        # 3. Estimate Uncertainty via MC Dropout (15 forward passes)
        mc_results = estimate_mc_uncertainty(self.model, xray_tensor, clinical_tensor, num_passes=15)
        
        # 4. Generate Explainability Heatmaps (Grad-CAM and Grad-CAM++)
        # Target layer is features.norm5, which is the final dense block norm layer
        target_layer = self.model.backbone.features.norm5
        
        # Grad-CAM explainer
        gcam = GradCAM(self.model, target_layer)
        heatmap_gcam = gcam.generate_heatmap(xray_tensor, clinical_tensor, target_class=1)
        gcam.remove_hooks()
        
        # Grad-CAM++ explainer
        gcam_plus = GradCAMPlusPlus(self.model, target_layer)
        heatmap_gcam_plus = gcam_plus.generate_heatmap(xray_tensor, clinical_tensor, target_class=1)
        gcam_plus.remove_hooks()
        
        # 5. Overlay heatmaps and save to disk
        gcam_path = os.path.join(output_vis_dir, f"{visit_id}_gradcam.png")
        gcam_plus_path = os.path.join(output_vis_dir, f"{visit_id}_gradcam_plus.png")
        
        save_xai_visualization(xray_tensor[0], heatmap_gcam, gcam_path)
        save_xai_visualization(xray_tensor[0], heatmap_gcam_plus, gcam_plus_path)
        
        total_latency = time.time() - start_time
        
        return {
            'predicted_class': mc_results['predicted_class'],
            'pneumonia_probability': mc_results['mean_prob'],
            'confidence': mc_results['confidence'],
            'uncertainty': mc_results['uncertainty'],
            'core_inference_latency': core_latency,
            'total_cdss_latency': total_latency,
            'gradcam_heatmap_path': gcam_path,
            'gradcam_plus_heatmap_path': gcam_plus_path,
            'raw_probs': mc_results['raw_probs']
        }
