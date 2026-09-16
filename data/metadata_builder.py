import os
import pandas as pd

def build_final_registry(split_csv, clinical_csv, partition_csv, output_csv):
    split_df = pd.read_csv(split_csv)
    clinical_df = pd.read_csv(clinical_csv)
    
    # 1. Merge splits with clinical attributes on patient_id
    merged_df = pd.merge(split_df, clinical_df, on='patient_id', how='left')
    
    # 2. Merge with partition info for training samples
    # The partition_csv only contains train split images with client_id assignments.
    # Let's map client_id back to train split images and set client_id = -1 for val and test splits.
    merged_df['client_id'] = -1
    
    if os.path.exists(partition_csv):
        part_df = pd.read_csv(partition_csv)
        # Create a mapping of filepath -> client_id
        path_to_client = dict(zip(part_df['filepath'], part_df['client_id']))
        
        # Apply the mapping
        def get_client(row):
            if row['new_split'] == 'train':
                return path_to_client.get(row['filepath'], -1)
            return -1
            
        merged_df['client_id'] = merged_df.apply(get_client, axis=1)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Final consolidated metadata registry built and saved to: {output_csv}")
    print(f"Total entries: {len(merged_df)}")
    print(f"Train samples partitioned to clients: {len(merged_df[merged_df['client_id'] != -1])}")
    print(f"Validation samples: {len(merged_df[merged_df['new_split'] == 'val'])}")
    print(f"Test samples: {len(merged_df[merged_df['new_split'] == 'test'])}")

if __name__ == '__main__':
    split_path = os.path.join('reports', 'patient_split_metadata.csv')
    clinical_path = os.path.join('reports', 'synthetic_patients_clinical.csv')
    partition_path = os.path.join('reports', 'dirichlet_partitions.csv')
    final_output = os.path.join('reports', 'final_metadata_registry.csv')
    build_final_registry(split_path, clinical_path, partition_path, final_output)
