# Research Limitations & Clinical Disclaimers

This document highlights critical research limitations and compliance boundaries of the HydroFed-ICAF system.

---

## 1. Core Limitations
1. **Pediatric Focus**: The chest X-ray dataset is comprised exclusively of pediatric scans (ages 1 to 5 years). Model parameters do not represent adult lungs and cannot be transferred to general diagnostic clinics without validation on adult datasets.
2. **Synthetic Clinical Attributes**: All patient clinical attributes (Age, Gender, Diabetes, Smoke Exposure, Family History) are synthetically generated. They do NOT represent real patient clinical measurements.
3. **No Medical Correlation Claim**: The synthetic variables do not establish actual clinical relationships or risk factors. They are strictly intended for engineering testing of the cross-attention (AICA) architecture.
4. **Simulated Clinics**: The 60 clinics modeled in this system are simulated nodes. They do not represent real hospital cohorts.
5. **Simulated FL Environments**: The network communications run locally in a simulation loop and do not replicate latency, dropped packets, or routing complexities of real hospital LAN/WAN setups.
6. **Reverse Engineering Vulnerability**: AES-256-GCM secures parameter updates during network transit and storage. However, it does not prevent reverse-engineering of the model structure or weights if the local edge device is physically compromised.
7. **XAI Interpretability Warning**: Grad-CAM and Grad-CAM++ saliency heatmaps indicate layer activations only. They do not constitute diagnostic proof or prove actual anatomical disease consolidation.
8. **Longitudinal Disclaimer**: AI-assisted trend assessments (IMPROVING / WORSENING) indicate statistical score movements and do not establish clinical cure or recovery.
9. **Clinical Validation**: This system has not undergone clinical trials, clinical validation, or FDA clearance. It is a research prototype.
10. **Regulatory Requirements**: Any real-world clinical deployment requires strict validation, institutional review board (IRB) approval, compliance with HIPAA/GDPR privacy controls, security assessments, and supervision by licensed radiologists.
