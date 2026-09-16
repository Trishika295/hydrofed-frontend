import torch
import torch.nn as nn

class CentralizedFedProx:
    def __init__(self, mu=0.01):
        self.mu = mu

    def aggregate(self, client_nodes, client_sample_sizes):
        """Standard sample-weighted averaging, identical to FedAvg."""
        total_samples = sum(client_sample_sizes)
        if total_samples <= 0:
            return
            
        ref_state = client_nodes[0].get_model_parameters()
        global_state = {}
        
        for k in ref_state.keys():
            if ref_state[k].is_floating_point():
                weighted_sum = torch.zeros_like(ref_state[k])
                for idx, node in enumerate(client_nodes):
                    node_state = node.get_model_parameters()
                    weight = client_sample_sizes[idx] / total_samples
                    weighted_sum += weight * node_state[k]
                global_state[k] = weighted_sum
            else:
                global_state[k] = ref_state[k].clone()
                
        for node in client_nodes:
            node.load_model_parameters(global_state)
            
        return global_state

    def local_train_prox(self, node, dataloader, global_state, epochs=1, device='cpu'):
        """Local training loop with Proximal penalty: Loss = CE + (mu/2) * ||W - W_global||^2."""
        node.model.train()
        device = torch.device(device)
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for xray, clinical, label, _ in dataloader:
                xray, clinical, label = xray.to(device), clinical.to(device), label.to(device)
                
                node.optimizer.zero_grad()
                logits = node.model(xray, clinical)
                
                # Standard Loss
                loss_ce = node.criterion(logits, label)
                
                # Proximal Penalty
                prox_penalty = 0.0
                for name, param in node.model.named_parameters():
                    if name in global_state:
                        g_param = global_state[name].to(device)
                        prox_penalty += torch.sum((param - g_param) ** 2)
                        
                loss = loss_ce + (self.mu / 2.0) * prox_penalty
                loss.backward()
                
                nn.utils.clip_grad_norm_(node.model.parameters(), max_norm=1.0)
                node.optimizer.step()
                
                total_loss += loss.item() * xray.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total += xray.size(0)
                
        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return avg_loss, accuracy
