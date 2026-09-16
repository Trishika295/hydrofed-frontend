import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Cleans up hooks from the model."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_heatmap(self, xray_tensor, clinical_tensor, target_class=1):
        """
        Generates a Grad-CAM heatmap for a target class.
        """
        self.model.zero_grad()
        
        # Ensure input tensor requires gradients
        xray_tensor.requires_grad = True
        
        # Temporarily enable requires_grad on backbone parameters for hook capture
        original_requires_grad = []
        for param in self.model.backbone.parameters():
            original_requires_grad.append(param.requires_grad)
            param.requires_grad = True
            
        try:
            # Forward pass
            logits = self.model(xray_tensor, clinical_tensor)
            
            # Determine target score
            score = logits[0, target_class]
            
            # Backward pass
            score.backward(retain_graph=True)
        finally:
            # Restore original backbone requires_grad state
            for param, state in zip(self.model.backbone.parameters(), original_requires_grad):
                param.requires_grad = state
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Verify target layer.")
            
        # Get gradients and activations
        gradients = self.gradients[0]  # shape: (C, H, W)
        activations = self.activations[0]  # shape: (C, H, W)
        
        # Channel-wise pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # shape: (C, 1, 1)
        
        # Linear combination
        cam = torch.sum(weights * activations, dim=0)  # shape: (H, W)
        
        # Apply ReLU to keep positive influence only
        cam = F.relu(cam)
        
        # Normalize between [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        # Resize to input image resolution (224, 224)
        cam = cam.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        cam = F.interpolate(cam, size=(xray_tensor.shape[2], xray_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        
        return cam
