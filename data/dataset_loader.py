import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from data.preprocessing import ChestXRayPreprocessor

class HydroFedDataset(Dataset):
    def __init__(self, registry_csv_path, split='train', client_id=-1, primary_task=True, target_size=(224, 224), use_clahe=True, feature_cache=None):
        """
        PyTorch Dataset for HydroFed-ICAF.
        primary_task: True for NORMAL vs PNEUMONIA (binary), False for NORMAL vs BACTERIA vs VIRUS (three-class)
        """
        self.df = pd.read_csv(registry_csv_path)
        
        # Filter based on split and client_id
        if client_id != -1:
            self.df = self.df[(self.df['new_split'] == 'train') & (self.df['client_id'] == client_id)]
        else:
            self.df = self.df[self.df['new_split'] == split]
            
        self.df = self.df.reset_index(drop=True)
        
        self.preprocessor = ChestXRayPreprocessor(target_size=target_size, use_clahe=use_clahe)
        self.primary_task = primary_task
        self.feature_cache = feature_cache  # Dictionary of filepath -> cached features if running pre-extracted

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        
        # 1. Load X-ray representation
        if self.feature_cache and filepath in self.feature_cache:
            # Load from pre-extracted feature maps (early, middle, deep semantic)
            xray_data = self.feature_cache[filepath]
        else:
            xray_data = self.preprocessor.preprocess_image_path(filepath)

        # 2. Encode clinical variables
        # continuous representation: age (let's normalize with simple min-max: assume age in [0, 80])
        age_normalized = float(row['age']) / 80.0
        
        # categorical representation: gender
        gender_val = 1.0 if row['gender'] == 'Female' else 0.0
        
        # binary representations
        diabetes_val = float(row['diabetes'])
        smoke_val = float(row['passive_smoke_exposure'])
        family_val = float(row['family_respiratory_history'])
        
        clinical_tensor = torch.tensor([age_normalized, gender_val, diabetes_val, smoke_val, family_val], dtype=torch.float32)

        # 3. Label mapping
        if self.primary_task:
            # NORMAL = 0, PNEUMONIA = 1
            label = 0 if row['class'] == 'NORMAL' else 1
        else:
            # NORMAL = 0, BACTERIA = 1, VIRUS = 2
            sub_cls = row['subclass']
            if sub_cls == 'NORMAL':
                label = 0
            elif sub_cls == 'BACTERIA':
                label = 1
            else:
                label = 2

        return xray_data, clinical_tensor, label, filepath

def get_dataloader(registry_csv_path, split='train', client_id=-1, primary_task=True, batch_size=32, shuffle=True, pin_memory=False, feature_cache=None):
    dataset = HydroFedDataset(
        registry_csv_path=registry_csv_path,
        split=split,
        client_id=client_id,
        primary_task=primary_task,
        feature_cache=feature_cache
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' or client_id != -1 else False,
        num_workers=0,  # keep 0 for Windows compatibility and thread safety
        pin_memory=pin_memory
    )
    return loader
