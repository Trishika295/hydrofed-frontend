import os
import pandas as pd
import numpy as np

def perform_patient_split(report_csv_path, output_csv_path, seed=42):
    df = pd.read_csv(report_csv_path)
    
    # 1. Remove duplicates
    # Let's read the report JSON to get duplicate list, or just use pandas drop_duplicates by calculating hashes
    # Or, we can just filter using the list of duplicates.
    # To keep it simple and robust, let's load reports/dataset_report.json to get duplicate filepaths to remove.
    report_json_path = report_csv_path.replace('.csv', '.json')
    duplicates_to_remove = set()
    if os.path.exists(report_json_path):
        import json
        with open(report_json_path, 'r') as f:
            stats = json.load(f)
            for item in stats.get('duplicates', []):
                duplicates_to_remove.add(item['duplicate'])
    
    print(f"Original records: {len(df)}")
    df = df[~df['filepath'].isin(duplicates_to_remove)].copy()
    print(f"Records after removing duplicates: {len(df)}")

    # 2. Extract unique patients and their classes/counts to perform a stratified patient split
    # Each patient might have normal, bacteria, or virus images.
    # We assign a patient's dominant class for stratification:
    # If a patient has any PNEUMONIA, they are PNEUMONIA. Specifically, we can check their images.
    patient_records = []
    grouped = df.groupby('patient_id')
    for patient_id, group in grouped:
        classes = group['class'].tolist()
        subclasses = group['subclass'].tolist()
        
        # Determine dominant class for stratification
        if 'PNEUMONIA' in classes:
            main_cls = 'PNEUMONIA'
            sub_cls = 'BACTERIA' if 'BACTERIA' in subclasses else 'VIRUS'
        else:
            main_cls = 'NORMAL'
            sub_cls = 'NORMAL'
            
        patient_records.append({
            'patient_id': patient_id,
            'image_count': len(group),
            'main_class': main_cls,
            'sub_class': sub_cls
        })
        
    patient_df = pd.DataFrame(patient_records)
    print(f"Total unique patients: {len(patient_df)}")

    # 3. Perform splitting (70% train, 15% val, 15% test) stratified by patient's class
    np.random.seed(seed)
    patient_df = patient_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    train_ids, val_ids, test_ids = [], [], []
    
    # Stratify by main class
    for cls in ['NORMAL', 'PNEUMONIA']:
        cls_patients = patient_df[patient_df['main_class'] == cls]['patient_id'].tolist()
        n = len(cls_patients)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        train_ids.extend(cls_patients[:n_train])
        val_ids.extend(cls_patients[n_train:n_train+n_val])
        test_ids.extend(cls_patients[n_train+n_val:])
        
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    # Map back to original images
    def assign_split(pid):
        if pid in train_set:
            return 'train'
        elif pid in val_set:
            return 'val'
        elif pid in test_set:
            return 'test'
        return 'unknown'
        
    df['new_split'] = df['patient_id'].apply(assign_split)
    
    # Verify zero overlap
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    assert len(overlap_train_val) == 0, "Overlap between train and val!"
    assert len(overlap_train_test) == 0, "Overlap between train and test!"
    assert len(overlap_val_test) == 0, "Overlap between val and test!"
    
    print("\n=== Patient-Level Split Verification (Zero Overlap Guaranteed) ===")
    for s in ['train', 'val', 'test']:
        split_df = df[df['new_split'] == s]
        p_count = len(split_df['patient_id'].unique())
        img_count = len(split_df)
        normal_cnt = len(split_df[split_df['class'] == 'NORMAL'])
        pneu_cnt = len(split_df[split_df['class'] == 'PNEUMONIA'])
        bac_cnt = len(split_df[split_df['subclass'] == 'BACTERIA'])
        vir_cnt = len(split_df[split_df['subclass'] == 'VIRUS'])
        print(f"Split '{s.upper()}': {p_count} patients, {img_count} images. "
              f"NORMAL: {normal_cnt}, PNEUMONIA: {pneu_cnt} (BACTERIA: {bac_cnt}, VIRUS: {vir_cnt})")
              
    # Save the updated mapping
    df.to_csv(output_csv_path, index=False)
    print(f"\nPatient split metadata mapping written to: {output_csv_path}")

if __name__ == '__main__':
    report_csv = os.path.join('reports', 'dataset_report.csv')
    output_csv = os.path.join('reports', 'patient_split_metadata.csv')
    perform_patient_split(report_csv, output_csv)
