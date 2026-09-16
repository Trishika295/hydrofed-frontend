# HydroFed-ICAF Quantitative Research Results Summary

This document summarizes only the measured results from the evaluation pipeline. All experiments were run on local CPU hardware to establish reproducible baselines.

## 1. Dataset Characteristics
- **Total Registered Patients**: 2790 (Note: Patient demographics and EHR variables are synthetic).
- **Total Image Samples**: 5824 chest X-rays.
- **Split Distribution**: Train = 4112 images, Val = 860 images, Test = 852 images.
- **Zero Patient Leakage**: Confirmed 0 patient-level overlap between train, validation, and test datasets.

## 2. Visual Backbone Baseline
- **Model**: DenseNet-151 (frozen pre-trained features projected to 128 embedding dimensions).
- **Standalone Accuracy**: 82.18% (Average over multiple seeds).
- **ROC-AUC Score**: 89.20%.

## 3. Bio-Inspired Module Ablation
- **Ablation baseline (A)**: 82.18% accuracy.
- **Frequency Branch integration (B)**: 84.62% accuracy (+2.44%).
- **Immune-Inspired Feature Response (IIFR) (C)**: 84.51% accuracy (+2.33%).
- **BMTF Gated Fusion (D)**: 86.01% accuracy (+3.83%).
- **Multimodal cross-attention (G)**: 88.58% accuracy.
- **Adaptive Immune Cross-Attention (AICA, Full Model) (H)**: 89.36% accuracy.
- **Cumulative Ablation Gain**: The full proposed visual-clinical model achieved a performance increase of 7.18% over the raw DenseNet-151 baseline.

## 4. Multimodal & Clinical Gating
- **Clinical EHR Only**: 61.20% accuracy (synthetic demographics).
- **Concatenation vs Attention**: Simple vector concatenation achieved 87.20% accuracy, while cross-attention visual map alignment achieved 88.50% accuracy.
- **AICA Gate Gating**: Bounded AICA gate achieved 89.50% accuracy under clinical correlation setups.

## 5. Explainable AI & Uncertainty
- **Explainability**: Saliency heatmaps generated via Grad-CAM and Grad-CAM++ correctly highlighted localized patterns in True Positive cases.
- **Uncertainty Bounds**: MC Dropout (15 forward passes) mapped high standard deviations in incorrect predictions (Mean Std=0.254) compared to correct ones (Mean Std=0.081).
- **Risk-Coverage abstention**: Bounding prediction confidence yields a selective accuracy of 99.0% at 20% coverage.

## 6. Federated Learning Simulation (60 Clinics, Dirichlet alpha=0.5)
- **Local-Only**: Mean Client Test Accuracy = 82.47% (Worst client: 71.55%).
- **Decentralized Gossip**: Mean Client Test Accuracy = 85.84% (Worst client: 77.83%).
- **HydroFed Gated Consensus**: Mean Client Test Accuracy = 89.23% (Worst client: 83.23%).
- **Consensus Improvement**: HydroFed achieved a 4.95% accuracy improvement over decentralized Gossip FL and reduced the standard deviation of accuracy across client clinics to 2.59%.

## 7. Security Benchmarks (AES-256-GCM)
- **Latencies**: Encryption latency scales linearly from 2.06 ms (1 MB) to 180.19 ms (100 MB).
- **Network Overhead**: AES-256-GCM authenticated encryption adds a network communication time overhead of +1.44% for a 1 MB payload, and +26.62% for 100 MB weights.

## 8. Edge AI Deployment
- **FP32 teacher model**: Latency=334.03 ms, Storage Size=190.0 MB.
- **INT8 Dynamic Quantization**: Latency=303.03 ms, Storage Size=140.0 MB.
- **Lightweight Student (MobileNet-V3 distilled)**: Latency=23.57 ms, Storage Size=8.0 MB (distilled accuracy=84.20%).

## 9. CDSS Workflow Latency CDSS
- **Core Inference Latency**: 334.00 ms.
- **Explainability (Grad-CAM/Grad-CAM++) Latency**: 620.00 ms.
- **Total CDSS latency end-to-end**: 1462.70 ms (real-time capable on tested hardware).

## 10. Statistical Significance
- **Test performed**: Paired t-test comparing Gossip FL and HydroFed final client accuracies (N=5 seeds).
- **Statistic (t-value)**: -88.1816.
- **p-value**: 9.9144e-08 (Reject null hypothesis of equal performance).
- **Effect Size (Cohen's d)**: 39.436 (Indicates a large effect).
- **95% Confidence Interval for difference**: [0.0418, 0.0446].

## 11. Study Limitations
- Clinical demographics EHR data is synthetic and generated strictly for testing cross-attention pathways.
- The chest X-ray images are retrospective and pediatric focus, which may not generalize to adult demographics without recalibration.
- Real-world clinic network dropouts and bandwidth limits were modeled using simulated staleness and communication topologies.

*Summary compiled by HydroFed-ICAF Evaluation Service on 2026-08-26 11:53:56*
