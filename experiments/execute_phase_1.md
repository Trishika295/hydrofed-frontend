# HydroFed-ICAF Phase 1 Formula Execution & Intermediate Results

This document compiles the intermediate results of the three major formula groups implemented in the HydroFed-ICAF framework. 
The computations are performed using real visual and clinical data from the dataset (`final_metadata_registry.csv`) and real client performance registries (`fl_results.json`).

---

## 1. Cross-Attention & Gated AICA Modules

### Formula Formulations
1. **Dynamic Gate score**:
   $$G = \sigma(W_g \cdot [F_s \| F_f \| (F_s \odot F_f)])$$
   Where $F_s$ represents spatial visual features and $F_f$ represents Fourier frequency features.
   
2. **Visual Cross-Attention**:
   $$\text{Att}_{s \to f} = \text{Softmax}\left( \frac{Q_s K_f^T}{\sqrt{d}} \right) V_f$$
   $$\text{Att}_{f \to s} = \text{Softmax}\left( \frac{Q_f K_s^T}{\sqrt{d}} \right) V_s$$
   $$F_{\text{cross}} = \text{Att}_{s \to f} \| \text{Att}_{f \to s}$$
   
3. **Adaptive Fusion & LayerNorm**:
   $$F_{\text{fused}} = G \cdot \text{Att}_{s \to f} + (1 - G) \cdot \text{Att}_{f \to s}$$
   $$F_{\text{final}} = \text{LayerNorm}(F_{\text{fused}} + \text{AICA}(F_s, F_f))$$

### Measured Intermediate Tensor Statistics
Below is the table of the exact statistics (Shape, Mean, Standard Deviation, Minimum, Maximum) calculated during the execution of a real patient sample:

| Variable Name | Tensor Shape | Mean | Std Dev | Minimum | Maximum |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$Fs$** (Spatial features) | [1, 128, 7, 7] | -0.257206 | 1.985176 | -6.800609 | 5.800369 |
| **$Ff$** (Frequency features) | [1, 128, 7, 7] | -0.000033 | 0.111531 | -0.851724 | 1.080174 |
| **$Q_s$** (Spatial Query projection) | [1, 49, 128] | 0.121123 | 1.998103 | -7.831866 | 7.100307 |
| **$K_f$** (Frequency Key projection) | [1, 49, 128] | 0.002691 | 0.111499 | -1.071689 | 1.088322 |
| **$Att\_map\_sf$** (Attention weights) | [1, 49, 49] | 0.020408 | 0.003868 | 0.015913 | 0.065866 |
| **$Att\_sf$** (S-to-F attention output) | [1, 128, 7, 7] | -0.000540 | 0.101826 | -0.298190 | 0.252329 |
| **$Att\_fs$** (F-to-S attention output) | [1, 128, 7, 7] | 0.044323 | 1.796444 | -4.209751 | 4.560416 |
| **$F\_cross$** (Concat features) | [1, 256, 7, 7] | 0.021891 | 1.272464 | -4.209751 | 4.560416 |
| **$F\_aica$** (AICA projection) | [1, 128, 7, 7] | 0.121424 | 1.223577 | -3.025181 | 3.782410 |
| **$G$** (Gating factor values) | [1, 128, 7, 7] | 0.469090 | 0.265001 | 0.008631 | 0.995982 |
| **$F\_fused$** (Gated fused maps) | [1, 128, 7, 7] | 0.035207 | 1.098960 | -3.347507 | 3.384287 |
| **$F\_final$** (LayerNorm output) | [1, 128, 7, 7] | -0.000000 | 1.000078 | -3.227663 | 2.595977 |

### Intermediate Feature Activations Visualization
The channel-averaged activation maps of all intermediate layers are shown below:

![AICA Intermediate Feature Maps](file:///c:/Users/Soundarya/OneDrive/Desktop/HydroFed/experiments/phase_1_outputs/Fig_01_aica_intermediates.png)

---

## 2. HydroFed Pressure / Flow Federated Gating

### Formula Formulations
1. **Client Node Importance Score (Pressure)**:
   $$\text{Pressure}_i = \text{Pi} = \frac{\text{Accuracy}_i}{\text{CommCost}_i + \varepsilon}$$
   
2. **Consensus Aggregation Gated Weight**:
   $$W_{\text{global}} = \sum_{i=1}^N \left( \frac{\text{Pressure}_i}{\sum_{j=1}^N \text{Pressure}_j} \right) W_i$$

### Simulation Aggregation Weight Statistics
Using the real final model accuracies across the $N=60$ client clinics from `fl_results.json` and generated communication costs, we obtain:

- **Total Client Pressure Sum**: 21.734410
- **Average Node Pressure**: 0.362240 (Min: 0.188878, Max: 0.707058)
- **Average Aggregation Weight**: 1.667% (Min: 0.869%, Max: 3.253%)

### Clinic Pressures & Weights Distribution Plots
The client-by-client breakdown of the accuracies, communication costs, computed pressures, and global aggregation weights are shown in the distributions below:

![HydroFed Pressure & Aggregation Weight Distributions](file:///c:/Users/Soundarya/OneDrive/Desktop/HydroFed/experiments/phase_1_outputs/Fig_02_hydrofed_pressure.png)

### First 10 Client Nodes Breakdown
Here is the raw data calculated for the first 10 clinic nodes:

| Client ID | Accuracy | Communication Cost | Pressure | Global Agg Weight |
| :---: | :---: | :---: | :---: | :---: |
| **Client 00** | 87.92% | 2.5483 | 0.3450 | 1.587% |
| **Client 01** | 88.89% | 4.6226 | 0.1923 | 0.885% |
| **Client 02** | 85.01% | 3.8352 | 0.2217 | 1.020% |
| **Client 03** | 87.03% | 3.3552 | 0.2594 | 1.193% |
| **Client 04** | 88.75% | 1.7617 | 0.5038 | 2.318% |
| **Client 05** | 83.23% | 1.7616 | 0.4724 | 2.174% |
| **Client 06** | 89.67% | 1.4091 | 0.6363 | 2.928% |
| **Client 07** | 85.47% | 4.3182 | 0.1979 | 0.911% |
| **Client 08** | 90.36% | 3.3640 | 0.2686 | 1.236% |
| **Client 09** | 89.60% | 3.7491 | 0.2390 | 1.100% |

---
*Results computed and compiled by the HydroFed Phase 1 Evaluator on 2026-08-27.*
