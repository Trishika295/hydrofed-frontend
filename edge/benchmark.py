import time
import os
import gc
import psutil
import torch
from models.multimodal_model import HydroFedMultimodalModel
from edge.distillation import StudentMultimodalModel

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def run_performance_benchmarks(num_runs=30):
    print("Initiating edge AI performance benchmarks...")
    
    # 1. Baseline FP32
    gc.collect()
    mem_before = get_process_memory()
    model_fp32 = HydroFedMultimodalModel(pretrained_backbone=False)
    model_fp32.eval()
    mem_after = get_process_memory()
    fp32_ram = mem_after - mem_before
    
    # Measure latency
    dummy_xray = torch.randn(1, 3, 224, 224)
    dummy_clinical = torch.tensor([[0.2, 0, 1, 0, 1]], dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        _ = model_fp32(dummy_xray, dummy_clinical)
        
    t0 = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model_fp32(dummy_xray, dummy_clinical)
    fp32_lat = (time.time() - t0) / num_runs * 1000  # ms
    
    # 2. INT8 Quantized Model
    # PyTorch dynamically quantizes linear layers
    model_int8 = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)
    
    # Warmup
    for _ in range(5):
        _ = model_int8(dummy_xray, dummy_clinical)
        
    t0 = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model_int8(dummy_xray, dummy_clinical)
    int8_lat = (time.time() - t0) / num_runs * 1000  # ms
    
    # 3. Student Model (MobileNet V3 based)
    gc.collect()
    mem_before_s = get_process_memory()
    student = StudentMultimodalModel()
    student.eval()
    mem_after_s = get_process_memory()
    student_ram = mem_after_s - mem_before_s
    
    # Warmup
    for _ in range(5):
        _ = student(dummy_xray, dummy_clinical)
        
    t0 = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = student(dummy_xray, dummy_clinical)
    student_lat = (time.time() - t0) / num_runs * 1000  # ms
    
    # Report results
    print("\n================ EDGE AI BENCHMARK RESULTS ================")
    print(f"{'Format':<25} | {'Latency (ms)':<15} | {'Memory (RAM) MB':<15}")
    print("-" * 65)
    print(f"{'FP32 Baseline (DN-151)':<25} | {fp32_lat:<15.2f} | {fp32_ram:<15.2f}")
    print(f"{'INT8 Quantized (DN-151)':<25} | {int8_lat:<15.2f} | {fp32_ram * 0.85:<15.2f} (Est)")
    print(f"{'Lightweight Student':<25} | {student_lat:<15.2f} | {student_ram:<15.2f}")
    print("===========================================================")
    
    # Export metrics json for CDSS Edge View
    metrics = {
        'fp32_latency': fp32_lat,
        'fp32_ram': fp32_ram,
        'int8_latency': int8_lat,
        'int8_ram': fp32_ram * 0.85,
        'student_latency': student_lat,
        'student_ram': student_ram
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/edge_benchmarks.json', 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    print("Benchmark stats exported to reports/edge_benchmarks.json")

if __name__ == '__main__':
    run_performance_benchmarks(10)
