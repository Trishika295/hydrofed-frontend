import os
import pandas as pd
import numpy as np

class SyntheticClinicalGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_patient_attributes(self, patient_ids, mode='MODE_B'):
        """
        Generates clinical variables for a list of patient IDs.
        Modes:
          - MODE_A: Completely independent attributes.
          - MODE_B: Weak realistic correlations (e.g., older age correlates slightly with higher diabetes probability).
          - MODE_C: Controlled interaction experiment (correlates attributes slightly with hypothetical risk).
        """
        records = []
        for pid in patient_ids:
            # Deterministic generation using patient ID hash as seed offset to keep consistent
            pid_seed = int(hashlib.md5(pid.encode()).hexdigest(), 16) % (2**32)
            local_rng = np.random.default_rng(pid_seed)

            # Continuous representation: Age (centered around pediatric cases as Kaggle is a pediatric dataset,
            # but let's make it a general realistic mix, e.g. mean=6.0 years for pediatric focus, or 1 to 80 range.
            # Since Kaggle chest x-ray is pediatric, ages are usually 1-5 years.
            # Let's generate age: pediatric (1-10 years) with 80% probability, adult (11-80 years) with 20% probability.
            if local_rng.random() < 0.8:
                age = float(local_rng.uniform(0.5, 9.5))
            else:
                age = float(local_rng.uniform(10.0, 75.0))

            # Categorical representations
            gender = local_rng.choice(['Male', 'Female'])
            
            # Base rates
            p_diabetes = 0.08
            p_smoke = 0.25
            p_family = 0.15

            if mode == 'MODE_B':
                # Weak realistic correlations: older age -> higher diabetes
                if age > 40:
                    p_diabetes = 0.25
                elif age > 10:
                    p_diabetes = 0.10
                else:
                    p_diabetes = 0.01  # pediatric diabetes is rare

                # If family respiratory history is present, smoke exposure risk is slightly higher due to environment
                if local_rng.random() < 0.3:
                    p_smoke = 0.40

            elif mode == 'MODE_C':
                # Controlled interaction experiment: correlate variables slightly
                if age > 50 or age < 2:
                    p_smoke = 0.35
                p_diabetes = 0.15 if gender == 'Female' else 0.05

            diabetes = 1 if local_rng.random() < p_diabetes else 0
            smoke = 1 if local_rng.random() < p_smoke else 0
            family = 1 if local_rng.random() < p_family else 0

            records.append({
                'patient_id': pid,
                'age': round(age, 1),
                'gender': gender,
                'diabetes': diabetes,
                'passive_smoke_exposure': smoke,
                'family_respiratory_history': family,
                'data_type': 'synthetic',
                'synthetic': 1
            })

        return pd.DataFrame(records)

import hashlib

def run_generation(metadata_csv_path, output_csv_path, mode='MODE_B'):
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv_path}")

    df = pd.read_csv(metadata_csv_path)
    unique_patients = df['patient_id'].unique()
    print(f"Generating synthetic clinical features for {len(unique_patients)} patients...")

    gen = SyntheticClinicalGenerator(seed=42)
    clinical_df = gen.generate_patient_attributes(unique_patients, mode=mode)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    clinical_df.to_csv(output_csv_path, index=False)
    print(f"Synthetic clinical attributes saved to {output_csv_path}")
    print("\nSummary Statistics of Generated Synthetic Variables:")
    print(f"Average Age: {clinical_df['age'].mean():.2f} years")
    print(f"Gender Distribution:\n{clinical_df['gender'].value_counts(normalize=True)}")
    print(f"Diabetes Prevalence: {clinical_df['diabetes'].mean() * 100:.2f}%")
    print(f"Passive Smoke Exposure: {clinical_df['passive_smoke_exposure'].mean() * 100:.2f}%")
    print(f"Family Respiratory History: {clinical_df['family_respiratory_history'].mean() * 100:.2f}%")

if __name__ == '__main__':
    metadata_path = os.path.join('reports', 'patient_split_metadata.csv')
    output_path = os.path.join('reports', 'synthetic_patients_clinical.csv')
    run_generation(metadata_path, output_path)
