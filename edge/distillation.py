import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class StudentMultimodalModel(nn.Module):
    def __init__(self, embed_dim=64, num_classes=2, drop_rate=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Lightweight MobileNet-V3-Small visual backbone (fast CPU execution)
        mobilenet = models.mobilenet_v3_small(pretrained=False)
        self.backbone = mobilenet.features
        
        # Projection layer to map MobileNet features (576 channels at output stage) to embed_dim
        self.proj_visual = nn.Conv2d(576, embed_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Simple MLP Clinical Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Simplified Attention Fusion (concat + MLP)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classifier Head
        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, xray_img, clinical_vector):
        # Extract features
        x_feat = self.proj_visual(self.pool(self.backbone(xray_img)))  # [B, embed_dim, 7, 7]
        x_pooled = self.pool_global(x_feat).view(-1, self.embed_dim)    # [B, embed_dim]
        
        # Encode clinical features
        c_feat = self.clinical_encoder(clinical_vector)  # [B, embed_dim]
        
        # Fusion
        fused = torch.cat([x_pooled, c_feat], dim=1)  # [B, embed_dim * 2]
        representation = self.fusion(fused)           # [B, embed_dim]
        
        # Classification logits
        logits = self.classifier(representation)
        return logits

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, targets):
        """Computes knowledge distillation loss."""
        # Hard label loss
        loss_ce = self.ce_loss(student_logits, targets)
        
        # Soft label loss (KL Divergence on temperature scaled logits)
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        log_p_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        loss_kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Balanced loss
        loss_total = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kl
        return loss_total

if __name__ == '__main__':
    # Verify shapes
    student = StudentMultimodalModel()
    dummy_xray = torch.randn(2, 3, 224, 224)
    dummy_clinical = torch.tensor([[0.2, 0, 1, 0, 1], [0.8, 1, 0, 1, 0]], dtype=torch.float32)
    
    out = student(dummy_xray, dummy_clinical)
    print("Student model output logits shape:", out.shape)
