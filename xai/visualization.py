import cv2
import numpy as np
import torch

def overlay_heatmap(image_tensor, heatmap, alpha=0.6):
    """
    Overlays a 2D saliency heatmap onto a 3-channel image tensor.
    Args:
        image_tensor: Normalized image tensor of shape (3, H, W) or (1, H, W) or NumPy array
        heatmap: 2D NumPy array of shape (H, W) in [0, 1] range
        alpha: Weight of the original image (opacity)
    Returns:
        composite: Visualized RGB NumPy image of shape (H, W, 3) in [0, 255] uint8 range
    """
    # 1. Convert tensor to numpy array [0, 255] grayscale or RGB
    if isinstance(image_tensor, torch.Tensor):
        # Move channel dimension to the end: shape becomes (H, W, C)
        img_np = image_tensor.detach().cpu().numpy()

        # If it was normalized, we should unnormalize:
        # In preprocessing.py, we did: img_norm = (img_normalized - mean) / std
        # Let's simple min-max project it back to [0, 255] for visual correctness:
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        else:
            img_np = np.zeros_like(img_np)
        img_np = (img_np * 255.0).astype(np.uint8)
        img_np = np.transpose(img_np, (1, 2, 0))  # shape: (H, W, 3)
    else:
        # Assume it is already a numpy array [0, 255]
        img_np = image_tensor.copy().astype(np.uint8)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Convert to RGB if it is grayscale
    if img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 3:
        # OpenCV works with BGR, let's keep it BGR for cv2 overlay then convert to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Convert heatmap [0, 1] to [0, 255] and apply colormap
    heatmap_255 = (heatmap * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)

    # 3. Superimpose heatmap onto original image
    composite = cv2.addWeighted(img_np, alpha, heatmap_color, 1.0 - alpha, 0)
    
    # Convert BGR back to RGB for PIL / Matplotlib / Web browser displays
    composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    
    return composite

def save_xai_visualization(image_tensor, heatmap, output_path, alpha=0.6):
    """Generates the overlay and writes it directly to disk."""
    vis = overlay_heatmap(image_tensor, heatmap, alpha=alpha)
    # Save as BGR for cv2.imwrite
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, vis_bgr)
    return vis
