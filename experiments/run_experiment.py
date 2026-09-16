import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from data.dataset_loader import get_dataloader

from federated.clinic import LocalClinicNode
from federated.topology import generate_small_world_topology, generate_ring_topology, generate_grid_topology
from federated.hydrofed import HydroFedConsensusAggregator
from federated.gossip import DecentralizedGossip
from federated.fedavg import CentralizedFedAvg
from federated.fedprox import CentralizedFedProx
from security.aes_gcm import AESGCMEncryptor
from security.key_manager import PairwiseKeyManager
from security.authentication import ClinicAuthenticationService
from security.secure_update import SecureUpdateValidator

# Set fixed seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

class UnifiedExperimentRunner:
    def __init__(self, config_path='configs/base.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        set_seeds(self.config['security'].get('seed', 42))
        self.device = torch.device(self.config['training']['device'])
        
        # Load registry
        self.registry_csv = self.config['data']['registry_csv']
        
        # Generate topology
        self.num_clients = self.config['federated']['num_clients']
        self.topology_type = self.config['federated']['topology']
        
        if self.topology_type == 'small_world':
            self.adj = generate_small_world_topology(
                self.num_clients, 
                k=self.config['federated']['small_world_k'],
                p=self.config['federated']['small_world_p']
            )
        elif self.topology_type == 'grid':
            self.adj = generate_grid_topology(self.num_clients)
        else:
            self.adj = generate_ring_topology(self.num_clients)
            
        # Security interfaces
        self.key_manager = PairwiseKeyManager(self.config['security']['master_secret'])
        self.auth_service = ClinicAuthenticationService(range(self.num_clients))
        self.validator = SecureUpdateValidator(self.config['security']['max_norm_threshold'])
        
        # Instantiate a shared visual backbone to optimize CPU memory
        from models.densenet151 import DenseNet151Backbone
        self.shared_backbone = DenseNet151Backbone(pretrained=False)

    def run_decentralized_hydrofed(self, rounds=3):
        print("\n--- Running HydroFed Decentralized Simulation ---")
        
        # 1. Instantiate isolated clients
        clients = [
            LocalClinicNode(
                client_id=i, 
                registry_csv_path=self.registry_csv, 
                embed_dim=self.config['model']['embed_dim'],
                lr=self.config['training']['lr'],
                device=self.config['training']['device'],
                shared_backbone=self.shared_backbone
            ) for i in range(self.num_clients)
        ]
        
        # 2. Get client Dataloaders
        dataloaders = [node.get_local_loader(batch_size=self.config['training']['batch_size']) for node in clients]
        sample_sizes = [len(dl.dataset) for dl in dataloaders]
        
        aggregator = HydroFedConsensusAggregator(
            gamma=self.config['hydrofed']['flow_gamma'],
            eta_max=self.config['hydrofed']['eta_max']
        )
        
        consensus_errors = []
        
        for r in range(rounds):
            print(f"Round {r+1}/{rounds}: Training local client models...")
            
            # Step A: Local Training epochs
            for i, node in enumerate(clients):
                if sample_sizes[i] == 0:
                    continue
                # We limit local training to 3 batches on CPU to run instantly
                # This performs full backpropagation and optimization
                node.model.train()
                for idx, (xray, clinical, label, _) in enumerate(dataloaders[i]):
                    if idx >= 3: break  # CPU speedup
                    xray, clinical, label = xray.to(self.device), clinical.to(self.device), label.to(self.device)
                    node.optimizer.zero_grad()
                    logits = node.model(xray, clinical)
                    loss = node.criterion(logits, label)
                    loss.backward()
                    node.optimizer.step()

            # Step B: Secure Gossip Consensus Step
            # Pairwise gossip exchange over communication topology edges
            total_disagreement = 0.0
            exchanges_count = 0
            
            # Encrypted exchanges list
            for i in range(self.num_clients):
                neighbors = self.adj.get(i, [])
                for j in neighbors:
                    if j > i:  # Avoid duplicate undirected checks
                        node_i = clients[i]
                        node_j = clients[j]
                        
                        # 1. Secure handshake authentication
                        token_i = self.auth_service.get_auth_token(i)
                        token_j = self.auth_service.get_auth_token(j)
                        
                        # 2. Key derivation and Encrypt weights prior to sending
                        key = self.key_manager.get_pairwise_key(i, j)
                        enc_i = AESGCMEncryptor(key)
                        enc_j = AESGCMEncryptor(key)
                        
                        meta_i = f"sender={i};receiver={j};round={r}"
                        meta_j = f"sender={j};receiver={i};round={r}"
                        
                        state_i = node_i.get_model_parameters()
                        state_j = node_j.get_model_parameters()
                        
                        nonce_i, cipher_i = enc_i.encrypt_state_dict(state_i, meta_i)
                        nonce_j, cipher_j = enc_j.encrypt_state_dict(state_j, meta_j)
                        
                        # 3. Simulate transmission to neighbors queues
                        node_i.queue_incoming_update(j, (nonce_j, cipher_j, meta_j))
                        node_j.queue_incoming_update(i, (nonce_i, cipher_i, meta_i))
                        
                        # 4. Decrypt, Validate, and Perform dynamic water-flow aggregate
                        # Clinic i processes update from j
                        dec_j = enc_j.decrypt_state_dict(nonce_j, cipher_j, meta_j)
                        valid_j, err_j = self.validator.validate_update(state_i, dec_j)
                        
                        # Clinic j processes update from i
                        dec_i = enc_i.decrypt_state_dict(nonce_i, cipher_i, meta_i)
                        valid_i, err_i = self.validator.validate_update(state_j, dec_i)
                        
                        if valid_i and valid_j:
                            # Run consensus updates
                            disagreement, eta = aggregator.execute_pairwise_consensus(node_i, node_j)
                            total_disagreement += disagreement
                            exchanges_count += 1
                            
            avg_disagreement = total_disagreement / max(1, exchanges_count)
            consensus_errors.append(avg_disagreement)
            print(f"Round {r+1} finished. Avg pairwise model disagreement (Consensus Error): {avg_disagreement:.4f}")

        # 3. Evaluate models on Test dataset
        test_loader = get_dataloader(self.registry_csv, split='test', primary_task=True, batch_size=32, shuffle=False)
        test_metrics = self._evaluate_all_nodes(clients, test_loader)
        
        return test_metrics, consensus_errors

    def _evaluate_all_nodes(self, clients, test_loader):
        accuracies = []
        f1s = []
        recalls = []
        aucs = []
        
        y_true = []
        for _, _, label, _ in test_loader:
            y_true.extend(label.tolist())
            
        y_true = np.array(y_true)
        
        for node in clients:
            node.model.eval()
            y_pred = []
            y_prob = []
            
            with torch.no_grad():
                for xray, clinical, _, _ in test_loader:
                    xray, clinical = xray.to(self.device), clinical.to(self.device)
                    logits = node.model(xray, clinical)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    y_prob.extend(probs.tolist())
                    y_pred.extend(preds.tolist())
                    
            accuracies.append(accuracy_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_prob))
            
        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_f1': float(np.mean(f1s)),
            'mean_recall': float(np.mean(recalls)),
            'mean_auc': float(np.mean(aucs)),
            'worst_accuracy': float(np.min(accuracies)),
            'best_accuracy': float(np.max(accuracies))
        }

    def run_baselines(self, rounds=3):
        """Runs FedAvg and Gossip Baselines for quantitative research comparison."""
        print("\n--- Running FedAvg Baseline ---")
        clients = [
            LocalClinicNode(
                client_id=i, registry_csv_path=self.registry_csv, 
                embed_dim=self.config['model']['embed_dim'], device=self.config['training']['device'],
                shared_backbone=self.shared_backbone
            ) for i in range(self.num_clients)
        ]
        
        dataloaders = [node.get_local_loader(batch_size=self.config['training']['batch_size']) for node in clients]
        sample_sizes = [len(dl.dataset) for dl in dataloaders]
        
        fedavg = CentralizedFedAvg()
        
        for r in range(rounds):
            for i, node in enumerate(clients):
                if sample_sizes[i] == 0: continue
                node.model.train()
                for idx, (xray, clinical, label, _) in enumerate(dataloaders[i]):
                    if idx >= 3: break  # Speedup
                    xray, clinical, label = xray.to(self.device), clinical.to(self.device), label.to(self.device)
                    node.optimizer.zero_grad()
                    logits = node.model(xray, clinical)
                    loss = node.criterion(logits, label)
                    loss.backward()
                    node.optimizer.step()
                    
            # Central weight aggregation
            fedavg.aggregate(clients, sample_sizes)
            
        test_loader = get_dataloader(self.registry_csv, split='test', primary_task=True, batch_size=32, shuffle=False)
        fedavg_metrics = self._evaluate_all_nodes(clients, test_loader)
        
        # --- Run Gossip ---
        print("\n--- Running Gossip Decentralized Baseline ---")
        g_clients = [
            LocalClinicNode(
                client_id=i, registry_csv_path=self.registry_csv, 
                embed_dim=self.config['model']['embed_dim'], device=self.config['training']['device'],
                shared_backbone=self.shared_backbone
            ) for i in range(self.num_clients)
        ]
        gossip = DecentralizedGossip()
        
        for r in range(rounds):
            for i, node in enumerate(g_clients):
                if sample_sizes[i] == 0: continue
                node.model.train()
                for idx, (xray, clinical, label, _) in enumerate(dataloaders[i]):
                    if idx >= 3: break  # Speedup
                    xray, clinical, label = xray.to(self.device), clinical.to(self.device), label.to(self.device)
                    node.optimizer.zero_grad()
                    logits = node.model(xray, clinical)
                    loss = node.criterion(logits, label)
                    loss.backward()
                    node.optimizer.step()
            gossip.execute_gossip_averaging(g_clients, self.adj)
            
        gossip_metrics = self._evaluate_all_nodes(g_clients, test_loader)
        
        return fedavg_metrics, gossip_metrics

if __name__ == '__main__':
    runner = UnifiedExperimentRunner()
    
    # Run 3 rounds of simulation for CPU speed
    hydrofed_metrics, consensus_history = runner.run_decentralized_hydrofed(rounds=3)
    fedavg_metrics, gossip_metrics = runner.run_baselines(rounds=3)
    
    # Save compilation to disk
    results = {
        'hydrofed': hydrofed_metrics,
        'fedavg': fedavg_metrics,
        'gossip': gossip_metrics,
        'consensus_history': consensus_history
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n\n=== FINAL SIMULATION METRICS ===")
    print(f"HydroFed Mean Test Acc: {hydrofed_metrics['mean_accuracy'] * 100:.2f}% (Worst client: {hydrofed_metrics['worst_accuracy']*100:.2f}%)")
    print(f"FedAvg Mean Test Acc: {fedavg_metrics['mean_accuracy'] * 100:.2f}% (Worst client: {fedavg_metrics['worst_accuracy']*100:.2f}%)")
    print(f"Gossip Mean Test Acc: {gossip_metrics['mean_accuracy'] * 100:.2f}% (Worst client: {gossip_metrics['worst_accuracy']*100:.2f}%)")
