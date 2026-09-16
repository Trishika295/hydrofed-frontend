# HydroFed-ICAF

**HydroFed-ICAF** is a Bio-Inspired Multimodal Decentralized Federated Edge-AI Clinical Decision Support System (CDSS) for chest X-ray pneumonia detection and longitudinal patient monitoring.

---

## 1. Architecture Overview
HydroFed-ICAF integrates clinical context (Age, Gender, Diabetes, Smoke Exposure, Family History) with chest X-ray representations to improve pneumonia detection under non-IID clinic distributions.

```
                         HYDROFED-ICAF
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
   PATIENT/CDSS           EDGE AI              FEDERATED
        |                     |                  NETWORK
        |                     |                     |
 Patient ID              Preprocessing          60 Clinics
 Visit ID                DenseNet-151          α = 0.5
 Clinical Data           IIFR                   HydroFed
 X-ray                   BMTF                   Pressure Flow
 History                 Frequency              Async FL
 Trend                   AICA                   AES-GCM
 Clinician Review        XAI                    Secure Updates
        |                     |                     |
        +---------------------+---------------------+
                              |
                         LOCAL CLINIC
                              |
                    Patient Data NEVER Leaves
                              |
                              X
                              |
                       Federated Layer
                              |
                       Only encrypted
                        model updates
```

---

## 2. Key Modules
- **BMTF-IIFR visual branch**: DenseNet-151 + multi-scale features + Fourier frequency transformations + Immune-Inspired Feature Response selective channel amplification.
- **Adaptive Immune Cross-Attention (AICA)**: Transformer cross-attention gating demographics tokens relative to visual maps.
- **HydroFed Gossip Consensus**: Parameter pressure-driven gossip flow with update conservation.
- **Security & Privacy**: AES-256-GCM encrypted updates, signature checks, and shape/NaN validators.
- **Database & Dashboard**: Local SQLite DB with longitudinal engine and REST HTTP server dashboard.

---

## 3. Quick Start

### Installation
Install python dependencies:
```bash
pip install -r requirements.txt
```

### Running Unit & Integration Tests
Run the complete unit test suites:
```bash
python -m unittest tests/test_model.py
python -m unittest tests/test_security.py
python -m unittest tests/test_database.py
python -m unittest tests/test_federated.py
```

### Running CDSS Dashboard Portal
Start the clinician dashboard workspace (runs on port 8080 by default):
```bash
python -m cdss.dashboard
```
Alternatively, run Streamlit directly:
```bash
streamlit run cdss/app.py
```
Open your web browser and navigate to `http://localhost:8080` (or `http://localhost:8501` for direct streamlit run) to access the interactive clinicians dashboard workspace.


### Running Federated Simulation
To simulate the 60-node decentralized consensus and save baselines:
```bash
python -m experiments.run_experiment
```
Results will write to `results/experiment_results.json`.
