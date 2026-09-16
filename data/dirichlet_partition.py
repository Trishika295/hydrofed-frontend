import os
import json
import pandas as pd
import numpy as np

def dirichlet_partition_train_data(metadata_csv_path, output_csv_path, num_clients=60, alpha=0.5, seed=42):
    df = pd.read_csv(metadata_csv_path)
    train_df = df[df['new_split'] == 'train'].copy()
    
    # We partition by class labels: NORMAL (0), PNEUMONIA (1)
    # We can also sub-partition pneumonia into BACTERIA (1) and VIRUS (2) for multi-class,
    # but let's partition based on the 3 subclasses: NORMAL, BACTERIA, VIRUS to make it super robust!
    classes = sorted(train_df['subclass'].unique())  # ['BACTERIA', 'NORMAL', 'VIRUS']
    num_classes = len(classes)
    
    np.random.seed(seed)
    
    # Store indices of samples for each class
    class_indices = {cls: train_df[train_df['subclass'] == cls].index.values for cls in classes}
    
    # Dirichlet distribution for class allocations
    # shape: (num_classes, num_clients)
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Adjust proportions to ensure we allocate all samples
    client_indices = [[] for _ in range(num_clients)]
    
    for c_idx, cls in enumerate(classes):
        indices = np.copy(class_indices[cls])
        np.random.shuffle(indices)
        
        # Calculate sample counts per client for this class
        class_proportions = proportions[c_idx]
        # Normalize just in case
        class_proportions = class_proportions / class_proportions.sum()
        
        # Split indices
        split_points = (np.cumsum(class_proportions) * len(indices)).astype(int)
        split_points[-1] = len(indices)  # ensure we allocate everything
        
        start = 0
        for client_id in range(num_clients):
            end = split_points[client_id]
            client_indices[client_id].extend(indices[start:end])
            start = end
            
    # Assign client IDs back to dataframe
    train_df['client_id'] = -1
    for client_id in range(num_clients):
        idxs = client_indices[client_id]
        train_df.loc[idxs, 'client_id'] = client_id
        
    # Save the updated train metadata with client partition IDs
    train_df.to_csv(output_csv_path, index=False)
    print(f"Dirichlet partition completed. Assigned client IDs to {len(train_df)} training samples.")
    
    # Calculate statistics per client
    stats = {}
    for client_id in range(num_clients):
        client_df = train_df[train_df['client_id'] == client_id]
        total = len(client_df)
        counts = client_df['subclass'].value_counts().to_dict()
        stats[str(client_id)] = {
            'total': total,
            'class_proportions': {cls: counts.get(cls, 0) / max(1, total) for cls in classes},
            'class_counts': {cls: int(counts.get(cls, 0)) for cls in classes}
        }
        
    stats_json_path = output_csv_path.replace('.csv', '_stats.json')
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Partition statistics saved to: {stats_json_path}")
    
    # Print sample distribution preview
    print("\nClient Partition Preview (First 5 clients):")
    for client_id in range(5):
        c_stats = stats[str(client_id)]
        prop_str = ", ".join([f"{cls}: {c_stats['class_counts'].get(cls, 0)} ({c_stats['class_proportions'].get(cls, 0.0)*100:.1f}%)" for cls in classes])
        print(f"Client {client_id:02d}: Total={c_stats['total']}, {prop_str}")

if __name__ == '__main__':
    metadata = os.path.join('reports', 'patient_split_metadata.csv')
    output = os.path.join('reports', 'dirichlet_partitions.csv')
    dirichlet_partition_train_data(metadata, output, num_clients=60, alpha=0.5)
