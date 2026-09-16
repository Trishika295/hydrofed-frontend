import os
import torch
import torch.nn as nn
from models.multimodal_model import HydroFedMultimodalModel

def run_dynamic_quantization(output_dir='checkpoints/quantized'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Instantiate FP32 Baseline
    model_fp32 = HydroFedMultimodalModel(pretrained_backbone=False)
    
    # Save FP32 Model State Dict
    fp32_path = os.path.join(output_dir, 'model_fp32.pt')
    torch.save(model_fp32.state_dict(), fp32_path)
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)  # MB
    
    # 2. Perform INT8 Dynamic Quantization
    # PyTorch dynamically quantizes nn.Linear layers for CPU inference speedups
    print("Applying dynamic INT8 quantization on linear layers...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},  # target layers
        dtype=torch.qint8
    )
    
    # Save INT8 State Dict
    int8_path = os.path.join(output_dir, 'model_int8.pt')
    # Save the entire quantized model structure, as state_dict doesn't preserve quantization maps directly in raw format
    torch.save(model_int8, int8_path)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)  # MB
    
    print("\n=== Edge Quantization Report ===")
    print(f"Baseline FP32 Model Size: {fp32_size:.2f} MB")
    print(f"Quantized INT8 Model Size: {int8_size:.2f} MB")
    print(f"Memory reduction ratio: {fp32_size / max(0.1, int8_size):.2f}x")
    
    # 3. Quick inference check to ensure accuracy remains intact
    dummy_xray = torch.randn(1, 3, 224, 224)
    dummy_clinical = torch.tensor([[0.2, 0, 1, 0, 1]], dtype=torch.float32)
    
    with torch.no_grad():
        out_fp32 = model_fp32(dummy_xray, dummy_clinical)
        out_int8 = model_int8(dummy_xray, dummy_clinical)
        
    print("\nSanity Inference Checks:")
    print("FP32 Logits:", out_fp32)
    print("INT8 Logits:", out_int8)
    return fp32_size, int8_size

if __name__ == '__main__':
    run_dynamic_quantization()
