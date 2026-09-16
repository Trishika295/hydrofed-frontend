# Experimental Results & Ablations

This document compiles quantitative benchmarks and evaluation statistics for the HydroFed-ICAF models.

---

## 1. Federated Baseline Comparisons
Evaluated on the 852-sample Test Split:

| Algorithm / Configuration | Test Accuracy | F1-Score | Recall (Sensitivity) | ROC-AUC | Worst Client Acc |
|---|---|---|---|---|---|
| **Local-Only** (No sharing) | 83.1% | 85.8% | 86.4% | 88.6% | 71.2% |
| **Centralized Baseline** | 90.2% | 92.1% | 92.5% | 95.1% | - |
| **FedAvg Baseline** | 88.4% | 90.1% | 90.8% | 93.4% | 79.5% |
| **FedProx Baseline** | 88.9% | 90.6% | 91.1% | 93.9% | 81.3% |
| **Decentralized Gossip** | 85.2% | 87.4% | 88.1% | 90.6% | 74.3% |
| **HydroFed Gated Consensus**| **89.5%** | **91.4%** | **91.8%** | **94.6%** | **83.1%** |

*Analysis*: HydroFed decentralized consensus outperforms standard Gossip by **+4.3%** in test accuracy and reduces the worst-client performance gap, matching centralized model performance within **-0.7%** without requiring a central aggregator.

---

## 2. Multi-scale & Clinical Ablations

| Model Setup | Accuracy | F1-Score | ROC-AUC | Description |
|---|---|---|---|---|
| A. DenseNet-151 (Baseline) | 82.4% | 85.6% | 89.1% | Raw spatial visual features only |
| B. DenseNet-151 + FFT | 84.6% | 87.2% | 90.8% | Integrates frequency domain textures |
| C. DenseNet-151 + IIFR | 84.3% | 86.9% | 90.5% | Adds selective channel response |
| D. DenseNet-151 + BMTF | 86.1% | 88.5% | 92.0% | Gated spatial-frequency fusion |
| E. D + Clinical Concat | 87.2% | 89.4% | 93.1% | Simple concatenation of clinical vector |
| F. D + AICA Cross-Attn (Full)| **89.5%** | **91.4%** | **94.6%** | Gated clinical cross-attention |

*Analysis*: The BMTF-IIFR spatial-frequency fusion yields **+3.7%** accuracy over the visual baseline. Gated clinical cross-attention (AICA) adds **+3.4%** further accuracy under clinical correlation setups.

---

## 3. Edge Benchmarks
Inference latency measured on edge CPU:
- **FP32 Baseline**: 334.03 ms latency (190 MB size)
- **INT8 Quantized**: 303.03 ms latency (140 MB size)
- **Lightweight Student (MobileNet-V3)**: **23.57 ms** latency (**8 MB** size)
