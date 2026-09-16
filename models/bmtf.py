import torch
import torch.nn as nn

class BioMultiscaleThreatFusion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # Conv 1x1 layer to compute gating factors from concatenated spatial and frequency representations
        self.conv_gate = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, Fi, Ff):
        """
        Args:
            Fi: Spatial (Immune-Inspired) feature map of shape (B, embed_dim, H, W)
            Ff: Frequency feature map of shape (B, embed_dim, H, W)
        Returns:
            F_xray: Fused X-ray representation tensor of shape (B, embed_dim, H, W)
        """
        # 1. Concatenate features along the channel dimension
        concat_features = torch.cat([Fi, Ff], dim=1)  # Shape: (B, embed_dim * 2, H, W)
        
        # 2. Compute dynamic gate alpha (values in [0, 1] for each channel and pixel)
        alpha = torch.sigmoid(self.conv_gate(concat_features))  # Shape: (B, embed_dim, H, W)
        
        # 3. Perform weighted threat-aware fusion
        F_xray = alpha * Fi + (1.0 - alpha) * Ff
        
        return F_xray

if __name__ == '__main__':
    bmtf = BioMultiscaleThreatFusion(embed_dim=128)
    dummy_Fi = torch.randn(2, 128, 7, 7)
    dummy_Ff = torch.randn(2, 128, 7, 7)
    out = bmtf(dummy_Fi, dummy_Ff)
    print("BMTF output shape:", out.shape)
