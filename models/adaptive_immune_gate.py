import torch
import torch.nn as nn

class AdaptiveImmuneCrossAttentionGate(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # Conv 1x1 to calculate dynamic gating scores
        self.conv_gate = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, F_xray, F_ca):
        """
        Args:
            F_xray: Spatial-frequency visual representation from BMTF, shape (B, embed_dim, H, W)
            F_ca: Multimodal clinical-attention representation, shape (B, embed_dim, H, W)
        Returns:
            F_final: Final fused representation, shape (B, embed_dim, H, W)
        """
        # 1. Channel-wise concatenation
        concat_feats = torch.cat([F_xray, F_ca], dim=1)  # Shape: (B, embed_dim * 2, H, W)
        
        # 2. Compute dynamic gate scores
        G = torch.sigmoid(self.conv_gate(concat_feats))  # Shape: (B, embed_dim, H, W)
        
        # 3. Apply gated clinical representation to spatial representation
        F_final = F_xray + G * F_ca
        
        return F_final

if __name__ == '__main__':
    aica = AdaptiveImmuneCrossAttentionGate(embed_dim=128)
    dummy_xray = torch.randn(2, 128, 7, 7)
    dummy_ca = torch.randn(2, 128, 7, 7)
    out = aica(dummy_xray, dummy_ca)
    print("AICA output shape:", out.shape)
