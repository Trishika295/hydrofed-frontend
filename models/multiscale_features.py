import torch
import torch.nn as nn

class MultiScaleFeatureProjector(nn.Module):
    def __init__(self, embed_dim=128, target_grid=(7, 7)):
        super().__init__()
        self.target_grid = target_grid
        
        # Conv 1x1 projections to map input channels to common embed_dim
        self.proj_early = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.proj_mid = nn.Conv2d(1504, embed_dim, kernel_size=1)
        self.proj_deep = nn.Conv2d(1264, embed_dim, kernel_size=1)
        
        # Adaptive pooling to resize spatial dimensions to target grid
        self.pool = nn.AdaptiveAvgPool2d(target_grid)

    def forward(self, features):
        """
        Args:
            features: list of 3 tensors [early, middle, deep] from DenseNet-151
        Returns:
            Fs: combined spatial representation tensor of shape (B, embed_dim, 7, 7)
        """
        early, mid, deep = features
        
        # Project and pool each stage
        f_early_proj = self.pool(self.proj_early(early))
        f_mid_proj = self.pool(self.proj_mid(mid))
        f_deep_proj = self.pool(self.proj_deep(deep))
        
        # Fuse multi-scale maps by summation
        Fs = f_early_proj + f_mid_proj + f_deep_proj
        return Fs

if __name__ == '__main__':
    projector = MultiScaleFeatureProjector(embed_dim=128)
    early = torch.randn(2, 256, 56, 56)
    mid = torch.randn(2, 1504, 14, 14)
    deep = torch.randn(2, 1264, 7, 7)
    
    out = projector([early, mid, deep])
    print("Projected multi-scale feature shape:", out.shape)
