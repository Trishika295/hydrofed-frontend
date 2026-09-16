import torch
import torch.nn as nn

class FrequencyFeatureBranch(nn.Module):
    def __init__(self, embed_dim=128, target_grid=(7, 7)):
        super().__init__()
        self.target_grid = target_grid
        
        # Lightweight CNN to extract frequency-domain features from the magnitude spectrum
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 112, 57]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, 56, 29]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Projection layer to align channels to embed_dim and spatial resolution to target_grid
        self.pool = nn.AdaptiveAvgPool2d(target_grid)
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input chest X-ray tensor of shape (B, 3, 224, 224) (replicated grayscale)
        Returns:
            Ff: frequency representation tensor of shape (B, embed_dim, 7, 7)
        """
        # 1. Take single grayscale channel (all channels are identical)
        gray = x[:, :1, :, :]  # Shape: (B, 1, 224, 224)
        
        # 2. Perform 2D Real FFT
        # Output shape: (B, 1, 224, 113) of complex tensors
        fft_out = torch.fft.rfft2(gray, norm='ortho')
        
        # 3. Compute absolute amplitude spectrum
        magnitude = torch.abs(fft_out)  # Shape: (B, 1, 224, 113)
        
        # 4. Pass through lightweight CNN layers
        feat = self.features(magnitude)
        
        # 5. Project and pool to align with spatial embedding (B, embed_dim, 7, 7)
        Ff = self.proj(self.pool(feat))
        
        return Ff

if __name__ == '__main__':
    branch = FrequencyFeatureBranch(embed_dim=128)
    dummy_xray = torch.randn(2, 3, 224, 224)
    out = branch(dummy_xray)
    print("Frequency branch output shape:", out.shape)
