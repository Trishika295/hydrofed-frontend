import torch

class CentralizedFedAvg:
    def __init__(self):
        pass

    def aggregate(self, client_nodes, client_sample_sizes):
        """
        Aggregates parameters from all client nodes using sample-weighted averaging.
        client_nodes: list of LocalClinicNode objects
        client_sample_sizes: list of integers representing sample count per client
        """
        total_samples = sum(client_sample_sizes)
        if total_samples <= 0:
            return
            
        # Get first node state to get structure
        ref_state = client_nodes[0].get_model_parameters()
        global_state = {}
        
        for k in ref_state.keys():
            if ref_state[k].is_floating_point():
                # Weighted average
                weighted_sum = torch.zeros_like(ref_state[k])
                for idx, node in enumerate(client_nodes):
                    node_state = node.get_model_parameters()
                    weight = client_sample_sizes[idx] / total_samples
                    weighted_sum += weight * node_state[k]
                global_state[k] = weighted_sum
            else:
                # Keep tracking parameters from first client
                global_state[k] = ref_state[k].clone()
                
        # Broadcast global state back to all clients
        for node in client_nodes:
            node.load_model_parameters(global_state)
            
        return global_state
