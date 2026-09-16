import math
import torch
import torch.nn as nn

class MultimodalCrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Caching the last attention weights for explainability
        self.attention_weights = None

    def forward(self, F_xray, F_clinical):
        """
        Args:
            F_xray: Spatial feature map of shape (B, embed_dim, H, W) -> Query
            F_clinical: Clinical representation tokens of shape (B, N_tokens, embed_dim) -> Key & Value (N_tokens = 5)
        Returns:
            F_ca: Fused representation map of shape (B, embed_dim, H, W)
        """
        B, C, H, W = F_xray.shape
        N_tokens = F_clinical.shape[1]
        
        # 1. Flatten X-ray spatially and transpose: (B, H*W, C)
        # Query inputs are the spatial patches
        xray_flat = F_xray.view(B, C, H * W).transpose(1, 2)  # Shape: (B, 49, embed_dim)
        
        # 2. Project Q, K, V
        q = self.q_proj(xray_flat)      # Shape: (B, 49, embed_dim)
        k = self.k_proj(F_clinical)     # Shape: (B, 5, embed_dim)
        v = self.v_proj(F_clinical)     # Shape: (B, 5, embed_dim)
        
        # 3. Reshape for Multi-Head Attention: (B, num_heads, length, head_dim)
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Scaled Dot-Product Attention
        # scores shape: (B, num_heads, 49, 5)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # Cache attention weights (averaged over heads) for CDSS XAI views
        self.attention_weights = attn.detach().mean(dim=1)  # Shape: (B, 49, 5)
        
        # context shape: (B, num_heads, 49, head_dim)
        context = torch.matmul(attn, v)
        
        # 5. Concatenate heads and project out
        context = context.transpose(1, 2).contiguous().view(B, H * W, C)
        out = self.out_proj(context)  # Shape: (B, 49, embed_dim)
        
        # 6. Reshape back to spatial tensor: (B, embed_dim, H, W)
        F_ca = out.transpose(1, 2).view(B, C, H, W)
        
        return F_ca

if __name__ == '__main__':
    attn = MultimodalCrossAttention(embed_dim=128)
    dummy_xray = torch.randn(2, 128, 7, 7)
    dummy_clinical = torch.randn(2, 5, 128)
    out = attn(dummy_xray, dummy_clinical)
    print("Cross-attention output shape:", out.shape)
    print("Cached attention weights shape:", attn.attention_weights.shape)
