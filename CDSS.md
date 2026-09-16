# Clinical Decision Support System (CDSS)

This document describes the design, features, and safety boundaries of the clinician-facing portal.

---

## 1. Clinician Workflow Portal
The CDSS UI is designed as a single-page web dashboard with 16 functional tabs. It provides an intuitive, real-time diagnostic cockpit for edge clinicians:
1. **Home**: Overview metrics and research disclaimers.
2. **Patient Registration**: Enters demographics to create local EHR records.
3. **Patient Search**: Looks up patient records and visit history in the local SQLite DB.
4. **X-ray Analysis**: Configures test-case images, displays raw and padded inputs.
5. **Clinical Inputs**: Displays active variables (Age, Gender, Diabetes, Smoke, Family).
6. **CDSS**: Classification output card (probability, confidence, uncertainty, latency).
7. **Explainability**: Side-by-side display of Grad-CAM and Grad-CAM++ saliency overlays.
8. **Longitudinal View**: Renders SVG time-series graphs of probabilities over visits.
9. **Edge Performance**: Displays quantization benchmarks and memory footprints.
10. **Federated Network**: Shows the 60-node topology and consensus error rates.
11. **HydroFed Flow**: Concept visualization of water-flow parameters container levels.
12. **Security**: AES-256-GCM encryption stats and validator logs.
13. **Experiments**: Ablation studies and comparative baseline graphs.
14. **Clinic Analytics**: Demographic subgroup accuracy and sensitivity audits.
15. **Model Info**: Structural layer dimensions.
16. **System Audit**: Time-stamped logs of all diagnostics, registrations, and reviews.

---

## 2. Safety Boundaries & Clinical Responsibilities

> [!WARNING]
> **CDSS SAFETY GUIDELINES & CONSTRAINTS**
> 1. **Decision Support, Not Diagnostic Authority**: The system provides an AI-assisted diagnostic estimate. It is NOT a replacement for a radiologist or physician.
> 2. **No Autonomous Prescriptions**: The CDSS does not suggest medications, treatments, or dosages. All clinical pathways must be ordered manually by a licensed physician.
> 3. **Uncertainty Alert Boundary**: If the predictive uncertainty (from MC Dropout standard deviation) exceeds the safety threshold ($0.15$), the dashboard triggers a high-priority alert: `CLINICIAN REVIEW REQUIRED`.
> 4. **Review Signature**: All clinical reviews require a clinician reference signature (e.g. Dr. Roberts) and are written directly to the SQLite audit log.
