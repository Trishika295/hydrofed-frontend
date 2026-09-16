# Longitudinal Patient Monitoring & Trend Engine

This document details the visit-to-visit tracking pipeline and temporal risk engines.

---

## 1. Longitudinal Database Mapping
Every patient has a unique persistent pseudonym: `patient_id` (e.g. `P-2026-000001`).
Each clinical visit generates a unique record linked to the patient: `visit_id` (e.g. `P-2026-000001-V001`).
The visit record captures:
- Visit timestamp.
- Preprocessed chest X-ray path.
- Clinical demographic variables.
- Model classification (probability, confidence, uncertainty, model version).
- Clinician review notes.

---

## 2. Temporal Trend Classification Engine
The `LongitudinalTrendEngine` analyzes historical probability coordinates over time:
- **IMPROVING**: Probability decreases by $\ge 0.10$ from the previous visit.
- **WORSENING**: Probability increases by $\ge 0.10$ from the previous visit.
- **STABLE**: Probability changes remain within $[-0.10, 0.10]$ boundaries.
- **UNCERTAIN**: Triggered if:
  - There is only 1 baseline visit recorded (requires subsequent points).
  - The model version changed between visits (see safety check below).

---

## 3. Safety Controls

### A. Model Version Consistency Check
If the model version changed between visits (e.g., from `v1.0` to `v2.0` due to a federated update), comparing raw probabilities directly is unsafe. The engine flags this:
`Model version changed between visits. Longitudinal probability comparison unavailable due to model-version difference.`
The trend resets to `UNCERTAIN` until multiple visits under the new model version are recorded.

### B. High Uncertainty Tagging
If the latest visit's MC standard deviation exceeds $0.15$, the trend engine appends a high-priority caution:
`High prediction uncertainty - clinician review required.`

---

## 4. Research Demonstration Mode
The Kaggle dataset contains isolated chest X-ray scans and does not provide genuine longitudinal records. The longitudinal views on the dashboard run in **Research Demonstration Mode**. All simulated longitudinal charts display:
`SIMULATED LONGITUDINAL HISTORY — RESEARCH DEMONSTRATION`
This makes the prototype's timeline and SVG chart capabilities clear without implying actual patient tracking.
