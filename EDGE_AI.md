# Edge AI Optimization & Benchmarks

This document describes model optimization and CPU performance benchmarks for local edge deployment.

---

## 1. Edge-Inference Architecture
To guarantee patient data privacy, model inference is executed locally at the clinic node. Chest X-rays never leave the clinic's local sandbox:
1. **Raw X-Ray** is acquired at the local clinic.
2. **Preprocessor** validates files, applies CLAHE, aspect-ratio padding, and mean/std normalization.
3. **Local Neural Network** combines spatial features (DenseNet-151), frequency features (Fourier), and patient demographics (Clinical Encoder) using cross-attention (AICA).
4. **Uncertainty Engine** samples logits via MC Dropout.
5. **Grad-CAM / Grad-CAM++** generates saliency maps for pathology review.
6. **SQLite DB** writes logs, visit observation details, and inferences.

---

## 2. Model Quantization
We dynamically quantize `nn.Linear` layers from Float32 to Int8 to accelerate CPU execution:
- Reduces memory bandwidth and execution bottlenecks.
- Maintains relative classification thresholds.

---

## 3. Knowledge Distillation Student Model
For resource-constrained edge devices (e.g. mobile health stations with single-core processors), we distilled the master multimodal network into a lightweight student model:
- **Student Backbone**: MobileNet-V3-Small (lightweight feature extractor).
- **Embed Dimensions**: Projected to $D=64$ instead of $D=128$.
- **Training Objective**: Standard Cross-Entropy + KL-Divergence loss from the DenseNet-151 teacher model.

---

## 4. Benchmark Performance Metrics (CPU Only)
Measured on CPU (Intel/AMD multithreaded runtime environment):

| Optimization Format | Average Latency (ms) | Memory footprint (RAM) | Model File Size |
|---|---|---|---|
| **FP32 Baseline (DN-151)** | 334.03 ms | 57.72 MB | ~190 MB |
| **INT8 Quantized (DN-151)** | 303.03 ms | 49.06 MB (Est) | ~140 MB |
| **Lightweight Student (MobileNet-V3)** | **23.57 ms** | **9.69 MB** | **~8 MB** |

*Conclusion*: The distilled student model achieves a **14.2x speedup** and **5.9x RAM reduction**, making it highly viable for real-time diagnostics on low-power edge hardware.
