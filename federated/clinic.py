import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.multimodal_model import HydroFedMultimodalModel
from data.dataset_loader import get_dataloader
from database.schema import init_database

class LocalClinicNode:
    def __init__(self, client_id, registry_csv_path, db_dir='database/clinics', embed_dim=128, lr=1e-3, device='cpu', shared_backbone=None):
        self.client_id = client_id
        self.device = torch.device(device)
        self.registry_csv_path = registry_csv_path
        
        # Isolated Database
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, f"clinic_{client_id}.db")
        init_database(self.db_path)
        
        # Local Model & Optimizer
        self.model = HydroFedMultimodalModel(embed_dim=embed_dim, num_classes=2, pretrained_backbone=False, shared_backbone=shared_backbone)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Message Queues
        self.incoming_updates = {}  # neighbor_id -> encrypted_state_dict
        self.outgoing_queue = []    # Queue of updates to send when network is online
        
        # Offline operation flag
        self.online = True

    def get_local_loader(self, batch_size=16, shuffle=True, feature_cache=None):
        """Returns local training data loader containing only this clinic's partitioned samples."""
        return get_dataloader(
            registry_csv_path=self.registry_csv_path,
            split='train',
            client_id=self.client_id,
            batch_size=batch_size,
            shuffle=shuffle,
            feature_cache=feature_cache
        )

    def local_train_epoch(self, dataloader, epochs=1):
        """Executes local gradient descent updates on local patient files."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for xray, clinical, label, _ in dataloader:
                xray, clinical, label = xray.to(self.device), clinical.to(self.device), label.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(xray, clinical)
                loss = self.criterion(logits, label)
                loss.backward()
                
                # Gradient clipping for stabilization
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item() * xray.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total += xray.size(0)
                
        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return avg_loss, accuracy

    def get_model_parameters(self):
        """Extracts clone of current model parameters."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def load_model_parameters(self, state_dict):
        """Loads model parameters into the local network."""
        self.model.load_state_dict(state_dict)

    def queue_incoming_update(self, sender_id, encrypted_update):
        """Saves incoming update to queue."""
        self.incoming_updates[sender_id] = encrypted_update
