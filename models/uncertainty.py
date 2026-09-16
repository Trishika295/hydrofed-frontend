import torch
import numpy as np

def enable_dropout_inference(model):
    """Force dropout layers to remain active during evaluation for MC Dropout."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def estimate_mc_uncertainty(model, xray_tensor, clinical_tensor, num_passes=15):
    """
    Computes predictions and uncertainty using Monte Carlo Dropout.
    Args:
        model: Trained multimodal neural network
        xray_tensor: Preprocessed chest X-ray image (1, 3, 224, 224)
        clinical_tensor: Clinical demographics vector (1, 5)
        num_passes: Number of forward passes to sample from the model
    Returns:
        dict containing:
            mean_prob: Mean prediction probability (e.g. of pneumonia class)
            predicted_class: 0 (Normal) or 1 (Pneumonia) based on mean_prob
            confidence: Confidence score (1.0 - std_deviation * 2)
            uncertainty: Standard deviation of probability predictions (variance indicator)
            raw_probs: List of probabilities from all passes
    """
    model.eval()
    enable_dropout_inference(model)  # Keep dropout active!
    
    probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            # Forward pass outputs logits: shape (B, num_classes)
            outputs = model(xray_tensor, clinical_tensor)
            
            # Binary classification case
            if outputs.shape[1] == 2:
                # Apply softmax to get probability for Pneumonia (class 1)
                prob = torch.softmax(outputs, dim=1)[:, 1].item()
            else:
                # Multi-class case, we can return the probability vector or focus on max class
                prob = torch.softmax(outputs, dim=1).squeeze(0).tolist()
                
            probs.append(prob)
            
    # Calculate statistics
    if isinstance(probs[0], list):
        # Multi-class case (3-class)
        probs_arr = np.array(probs)  # Shape: (num_passes, 3)
        mean_probs = np.mean(probs_arr, axis=0)
        std_probs = np.std(probs_arr, axis=0)
        
        predicted_class = int(np.argmax(mean_probs))
        mean_prob = float(mean_probs[predicted_class])
        uncertainty = float(std_probs[predicted_class])
    else:
        # Binary case (NORMAL vs PNEUMONIA)
        mean_prob = float(np.mean(probs))
        uncertainty = float(np.std(probs))
        predicted_class = 1 if mean_prob >= 0.5 else 0
        # If normal, we report probability of NORMAL as the reference, but let's report PNEUMONIA probability
        # Let's keep the probability of class 1 (PNEUMONIA) as mean_prob
        # Confidence score bounded in [0, 1]
        
    confidence = float(1.0 - (uncertainty * 2.0))
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        'mean_prob': mean_prob,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'raw_probs': probs
    }
