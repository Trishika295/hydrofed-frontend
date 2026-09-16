# Security & Privacy Protocol

HydroFed-ICAF enforces a defense-in-depth security model to protect patient privacy and network parameter integrity.

---

## 1. Local-Only Patient Data Boundary
Patient chest X-ray images, visit logs, clinical observations, and clinician notes reside exclusively within the clinic's local SQLite database. No raw patient measurements, EHR features, or image pixels are ever transmitted during federated learning or model consensus.

---

## 2. Secure Model Update Exchanges

### A. AES-256-GCM Encryption
Model updates transmitted over the network are encrypted using AES-256-GCM (Galois/Counter Mode). This provides:
- **Confidentiality**: Parameters cannot be eavesdropped.
- **Integrity**: Any bit-flipping during transmission corrupts the ciphertext and fails authentication.
- **Authentication**: Using associated data (metadata mapping sender, receiver, and round) prevents update replay attacks.

### B. Secure Key Management
Symmetric keys are derived dynamically per neighbor connection using SHA-256 HKDF-like derivation based on a master KMS/HSM secret and sorted client IDs. No shared static keys are hard-coded on disk.

### C. Handshake Authentication
Before establishing a session, clinic nodes perform challenge-response token authentication. Token comparisons use constant-time operations to mitigate timing analysis attacks.

---

## 3. Secure Update Validation (Validator)
Decrypted update payloads are validated against the current model parameters before being aggregated:
1. **Structural Checks**: Verify that all parameters match expected keys exactly.
2. **Dimension Checks**: Verify that all tensor shapes match.
3. **Data Type Checks**: Verify float32 compatibility.
4. **Value Checks**: Scan all parameters for `NaN` or `Inf` to block gradient explosion exploits.
5. **Anomaly Checks**: Compute the total L2 update difference norm. If it exceeds `max_norm_threshold` (default 100.0), the update is rejected as a poisoning attack.
