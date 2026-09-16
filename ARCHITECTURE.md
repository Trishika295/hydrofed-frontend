# HydroFed-ICAF System Architecture

This document describes the multi-stage visual, clinical, and consensus pipelines implemented in HydroFed-ICAF.

---

## 1. Multimodal Diagnostic Flow

```
X-Ray Image
    │
    ▼
Preprocessing (Resize, CLAHE, Mean/Std Norm)
    │
    ├───────────────────────────────────────────────┐
    ▼                                               ▼
DenseNet-151 Backbone (Frozen)              2D Fast Fourier (RFFT2)
    │                                               │
    ▼                                               ▼
Multi-Scale Feature Projector               Lightweight CNN Projection
    │                                               │
    ▼ (Fs)                                          ▼ (Ff)
Immune-Inspired Response (IIFR)                     │
    │ (Fi = Fs * (1 + λ * T))                       │
    ▼                                               ▼
    └───────────────► BMTF Gating Layer ◄───────────┘
                            │
                            ▼ (F_xray = α * Fi + (1 - α) * Ff)
                   AICA Cross-Attention
                            ▲
                            │
                     Clinical Encoder ◄─── EHR attributes
                            │
                            ▼ (F_final = F_xray + G * F_ca)
                      Global Pooling
                            │
                            ▼
                    MC Dropout (15 Passes)
                            │
                            ▼
                 Diagnostic Probability
```

---

## 2. Gating Mathematics

### A. Immune-Inspired Feature Response (IIFR)
For intermediate spatial feature maps $F \in \mathbb{R}^{B \times D \times H \times W}$:
$$T = \text{sigmoid}(W_t F + b_t)$$
$$F_{\text{immune}} = F \odot (1.0 + \lambda \cdot T)$$
Where:
- $T$ represents the learned spatial-channel response score (computed via a Conv 1x1 layer).
- $\lambda$ is a learnable model parameter representing amplification scaling.

### B. Bio-inspired Multi-scale Threat-aware Fusion (BMTF)
Learns a dynamic channel-wise gating factor $\alpha \in \mathbb{R}^{B \times D \times H \times W}$:
$$\alpha = \text{sigmoid}(W_f [F_{\text{immune}}; F_{\text{freq}}] + b_f)$$
$$F_{\text{xray}} = \alpha \odot F_{\text{immune}} + (1.0 - \alpha) \odot F_{\text{freq}}$$
Where:
- $[F_{\text{immune}}; F_{\text{freq}}]$ is the channel concatenation.
- $\alpha$ dynamically balances spatial visual pathology features and frequency domain textures.

### C. Adaptive Immune Cross-Attention (AICA)
Clinical tokens are derived from normalized Age and embedded categorical variables.
Using X-ray features $F_{\text{xray}}$ as Query ($Q$) and Clinical Tokens as Key/Value ($K, V$), cross-attention outputs $F_{\text{CA}}$.
The insertion of clinical context is regulated by the AICA gate:
$$G = \text{sigmoid}(W_g [F_{\text{xray}}; F_{\text{CA}}] + b_g)$$
$$F_{\text{final}} = F_{\text{xray}} + G \odot F_{\text{CA}}$$
This prevents clinical observations from polluting visual pathology representations if EHR data is corrupted or irrelevant.
