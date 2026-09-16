import torch
import numpy as np

class HydroFedFlowController:
    def __init__(self, gamma=0.1, eta_max=0.45):
        self.gamma = gamma
        self.eta_max = eta_max

    def compute_model_disagreement(self, state_dict_i, state_dict_j, key_prefix='classifier'):
        """
        Computes L2 distance (model pressure) between selected layers of two models.
        Focusing on top layers (classifier and fusion layers) is fast and representative.
        """
        disagreement = 0.0
        keys_count = 0
        
        for k in state_dict_i.keys():
            # Calculate disagreement based on classifier/fusion weights to save CPU cycles
            if key_prefix in k or 'proj' in k or 'gate' in k:
                wi = state_dict_i[k].float()
                wj = state_dict_j[k].float()
                disagreement += torch.norm(wi - wj).item()
                keys_count += 1
                
        # If no key matched the prefix, calculate on all weights
        if keys_count == 0:
            for k in state_dict_i.keys():
                if state_dict_i[k].is_floating_point():
                    wi = state_dict_i[k].float()
                    wj = state_dict_j[k].float()
                    disagreement += torch.norm(wi - wj).item()
                    
        return disagreement

    def compute_flow_coefficient(self, disagreement, resistance=1.0):
        """
        Computes the adaptive flow coefficient eta_ij based on pressure and resistance.
        eta_ij = clip(gamma * (disagreement / resistance), 0, eta_max)
        """
        if resistance <= 0:
            resistance = 1.0
            
        pressure = disagreement / resistance
        eta = self.gamma * pressure
        
        # Clip to ensure stability: eta must remain in [0, eta_max]
        eta = min(max(0.0, eta), self.eta_max)
        return eta
