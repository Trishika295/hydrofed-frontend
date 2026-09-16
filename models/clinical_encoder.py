import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Continuous Age MLP projection
        self.age_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. Embedding layers for binary/categorical variables (each maps 0/1 to embed_dim)
        self.gender_emb = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        self.diabetes_emb = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        self.smoke_emb = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        self.family_emb = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        
        # 3. Final projection
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, clinical_inputs):
        """
        Args:
            clinical_inputs: Tensor of shape (B, 5) where elements are:
              - [:, 0]: normalized age (0 to 1)
              - [:, 1]: gender (0 or 1)
              - [:, 2]: diabetes (0 or 1)
              - [:, 3]: smoke exposure (0 or 1)
              - [:, 4]: family respiratory history (0 or 1)
        Returns:
            clinical_representation: Tensor of shape (B, 5, embed_dim) representing clinical tokens
        """
        B = clinical_inputs.shape[0]
        
        # 1. Project Age
        age_in = clinical_inputs[:, 0].unsqueeze(1)  # Shape: (B, 1)
        age_token = self.age_mlp(age_in).unsqueeze(1)  # Shape: (B, 1, embed_dim)
        
        # 2. Embed Categorical Features (cast values to long for PyTorch embedding lookup)
        gender_in = clinical_inputs[:, 1].long()
        gender_token = self.gender_emb(gender_in).unsqueeze(1)  # Shape: (B, 1, embed_dim)
        
        diabetes_in = clinical_inputs[:, 2].long()
        diabetes_token = self.diabetes_emb(diabetes_in).unsqueeze(1)  # Shape: (B, 1, embed_dim)
        
        smoke_in = clinical_inputs[:, 3].long()
        smoke_token = self.smoke_emb(smoke_in).unsqueeze(1)  # Shape: (B, 1, embed_dim)
        
        family_in = clinical_inputs[:, 4].long()
        family_token = self.family_emb(family_in).unsqueeze(1)  # Shape: (B, 1, embed_dim)
        
        # 3. Concatenate tokens to get the clinical tokens set
        clinical_tokens = torch.cat([age_token, gender_token, diabetes_token, smoke_token, family_token], dim=1)  # Shape: (B, 5, embed_dim)
        
        # 4. Final normalization projection
        clinical_representation = self.norm(clinical_tokens)
        
        return clinical_representation

if __name__ == '__main__':
    encoder = ClinicalEncoder(embed_dim=128)
    dummy_clinical = torch.tensor([[0.1, 0, 1, 0, 1], [0.6, 1, 0, 1, 0]], dtype=torch.float32)
    out = encoder(dummy_clinical)
    print("Clinical tokens shape:", out.shape)
