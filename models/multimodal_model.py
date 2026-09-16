import torch
import torch.nn as nn
from models.densenet151 import DenseNet151Backbone
from models.multiscale_features import MultiScaleFeatureProjector
from models.frequency_branch import FrequencyFeatureBranch
from models.immune_feature_response import ImmuneFeatureResponse
from models.bmtf import BioMultiscaleThreatFusion
from models.clinical_encoder import ClinicalEncoder
from models.cross_attention import MultimodalCrossAttention
from models.adaptive_immune_gate import AdaptiveImmuneCrossAttentionGate

class HydroFedMultimodalModel(nn.Module):
    def __init__(self, embed_dim=128, num_classes=2, pretrained_backbone=True, drop_rate=0.3, shared_backbone=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # 1. Visual Backbone
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = DenseNet151Backbone(pretrained=pretrained_backbone)
        self.projector = MultiScaleFeatureProjector(embed_dim=embed_dim)
        
        # 2. Frequency Branch
        self.frequency_branch = FrequencyFeatureBranch(embed_dim=embed_dim)
        
        # 3. Immune-Inspired Feature Response
        self.iifr = ImmuneFeatureResponse(embed_dim=embed_dim)
        
        # 4. Bio-inspired Multi-scale Threat Fusion (BMTF)
        self.bmtf = BioMultiscaleThreatFusion(embed_dim=embed_dim)
        
        # 5. Clinical Demographic Encoder
        self.clinical_encoder = ClinicalEncoder(embed_dim=embed_dim)
        
        # 6. Adaptive Cross-Attention
        self.cross_attn = MultimodalCrossAttention(embed_dim=embed_dim)
        self.aica_gate = AdaptiveImmuneCrossAttentionGate(embed_dim=embed_dim)
        
        # 7. Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, xray_img, clinical_vector):
        """
        Args:
            xray_img: Preprocessed chest X-ray tensor of shape (B, 3, 224, 224)
            clinical_vector: Clinical demographic vector of shape (B, 5)
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        # Step A: Extract spatial multi-scale features from DenseNet-151
        if xray_img.requires_grad or self.training:
            raw_features = self.backbone(xray_img)
        else:
            with torch.no_grad():
                raw_features = self.backbone(xray_img)
            
        Fs = self.projector(raw_features)  # Shape: (B, embed_dim, 7, 7)
        
        # Step B: Apply Immune-Inspired Feature Response (IIFR) to spatial features
        Fi = self.iifr(Fs)  # Shape: (B, embed_dim, 7, 7)
        
        # Step C: Extract frequency-domain representation
        Ff = self.frequency_branch(xray_img)  # Shape: (B, embed_dim, 7, 7)
        
        # Step D: Apply Bio-inspired Multi-scale Threat-aware Fusion (BMTF)
        # Combine spatial-immune features and frequency features
        F_xray = self.bmtf(Fi, Ff)  # Shape: (B, embed_dim, 7, 7)
        
        # Step E: Encode patient demographics
        F_clinical = self.clinical_encoder(clinical_vector)  # Shape: (B, 5, embed_dim)
        
        # Step F: Apply Multimodal Cross-Attention
        F_ca = self.cross_attn(F_xray, F_clinical)  # Shape: (B, embed_dim, 7, 7)
        
        # Step G: Apply Adaptive Gate (AICA)
        F_final = self.aica_gate(F_xray, F_ca)  # Shape: (B, embed_dim, 7, 7)
        
        # Step H: Pooling & Classification
        pooled = self.pool(F_final).view(-1, self.embed_dim)  # Shape: (B, embed_dim)
        feat_drop = self.dropout(pooled)
        logits = self.classifier(feat_drop)  # Shape: (B, num_classes)
        
        return logits

if __name__ == '__main__':
    model = HydroFedMultimodalModel(pretrained_backbone=False)
    dummy_xray = torch.randn(2, 3, 224, 224)
    dummy_clinical = torch.tensor([[0.2, 0, 1, 0, 1], [0.8, 1, 0, 1, 0]], dtype=torch.float32)
    
    logits = model(dummy_xray, dummy_clinical)
    print("Multimodal output logits shape:", logits.shape)
