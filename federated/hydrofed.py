import torch
from federated.flow_controller import HydroFedFlowController

class HydroFedConsensusAggregator:
    def __init__(self, gamma=0.1, eta_max=0.45):
        self.flow_controller = HydroFedFlowController(gamma=gamma, eta_max=eta_max)

    def execute_pairwise_consensus(self, node_i, node_j, resistance=1.0):
        """
        Executes a symmetric water-flow consensus exchange between two neighbor nodes.
        Preserves model parameter conservation:
        Delta = eta * (W_j - W_i)
        W_i' = W_i + Delta
        W_j' = W_j - Delta
        """
        state_i = node_i.get_model_parameters()
        state_j = node_j.get_model_parameters()
        
        # 1. Compute pressure/disagreement
        disagreement = self.flow_controller.compute_model_disagreement(state_i, state_j)
        
        # 2. Compute adaptive flow rate eta_ij
        eta = self.flow_controller.compute_flow_coefficient(disagreement, resistance=resistance)
        
        if eta <= 0:
            return 0.0, 0.0  # No flow
            
        # 3. Perform symmetric parameter conservation updates
        new_state_i = {}
        new_state_j = {}
        
        for k in state_i.keys():
            wi = state_i[k]
            wj = state_j[k]
            
            # Exchanged only floating point parameter weights (avoid discrete buffers)
            if wi.is_floating_point():
                # Delta = eta * (W_j - W_i)
                delta = eta * (wj - wi)
                
                new_state_i[k] = wi + delta
                new_state_j[k] = wj - delta
            else:
                # Keep original non-float tracking parameters
                new_state_i[k] = wi.clone()
                new_state_j[k] = wj.clone()
                
        # Load states back
        node_i.load_model_parameters(new_state_i)
        node_j.load_model_parameters(new_state_j)
        
        return disagreement, eta
