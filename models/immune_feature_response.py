import torch
import torch.nn as nn

class ImmuneFeatureResponse(nn.Module):
    def __init__(self, embed_dim=128, init_lambda=0.1):
        super().__init__()
        # Conv 1x1 to learn the feature response scores (T)
        self.conv_t = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # Learnable amplification scale parameter
        self.lambda_param = nn.Parameter(torch.tensor(init_lambda))

    def forward(self, F):
        """
        Args:
            F: Input spatial feature map of shape (B, embed_dim, H, W)
        Returns:
            F_immune: Amplified immune-inspired response feature map of shape (B, embed_dim, H, W)
        """
        # Calculate response scores (values in [0, 1])
        T = torch.sigmoid(self.conv_t(F))
        
        # Apply response amplification
        # F_immune = F * (1 + lambda * T)
        F_immune = F * (1.0 + self.lambda_param * T)
        
        return F_immune

if __name__ == '__main__':
    iifr = ImmuneFeatureResponse(embed_dim=128)
    dummy_F = torch.randn(2, 128, 7, 7)
    out = iifr(dummy_F)
    print("IIFR output shape:", out.shape)
    print("Learnable lambda parameter:", iifr.lambda_param.item())
