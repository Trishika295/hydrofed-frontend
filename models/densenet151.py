import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.densenet import DenseNet

class DenseNet151Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # DenseNet-151 configuration: 1 + 1 + 2*(6 + 12 + 39 + 16) + 3 = 151 layers
        # Growth rate is 32 (same as DenseNet-121, 169, 201)
        self.densenet = DenseNet(growth_rate=32, block_config=(6, 12, 39, 16))
        
        if pretrained:
            self._load_pretrained_weights()
            
        # Extract features sub-blocks
        self.features = self.densenet.features
        
        # Freeze backbone weights to prevent gradient calculations
        for param in self.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self):
        """Loads weights from pretrained DenseNet-169 and maps matching layers."""
        try:
            # Load pretrained DenseNet-169 as it shares growth_rate=32 and first few blocks
            d169 = models.densenet169(pretrained=True)
            src_state = d169.state_dict()
            dest_state = self.densenet.state_dict()
            
            mapped_count = 0
            skipped_count = 0
            
            for name, param in src_state.items():
                if name in dest_state:
                    if param.shape == dest_state[name].shape:
                        dest_state[name].copy_(param)
                        mapped_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
                    
            self.densenet.load_state_dict(dest_state)
            print(f"Weight mapping complete. Successfully mapped {mapped_count} tensors from DenseNet-169. Skipped {skipped_count} unmatched tensors.")
        except Exception as e:
            print(f"Warning: Failed to map pretrained weights with error '{e}'. Using random initialization.")

    def forward(self, x):
        """
        Forward pass that returns feature maps at three different stages:
        - Early (Stage 1): output of denseblock1 (channels=256)
        - Middle (Stage 2): output of denseblock3 (channels=1504)
        - Deep (Stage 3): output of final features block (features.norm5, channels=1984)
        """
        features_list = []
        
        # Step through the DenseNet features sequentially
        out = x
        for name, layer in self.features.named_children():
            out = layer(out)
            if name == 'denseblock1':
                features_list.append(out)  # Early features
            elif name == 'denseblock3':
                features_list.append(out)  # Middle features
                
        features_list.append(out)  # Deep features (norm5 output)
        
        return features_list  # [early, middle, deep]

if __name__ == '__main__':
    model = DenseNet151Backbone(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)
    print("Stage 1 shape:", outputs[0].shape)
    print("Stage 2 shape:", outputs[1].shape)
    print("Stage 3 shape:", outputs[2].shape)
