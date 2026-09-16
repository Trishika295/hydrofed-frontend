import numpy as np

class AsynchronousHydroFedController:
    def __init__(self, beta=0.2):
        """
        beta: Decay coefficient for staleness calculation.
        """
        self.beta = beta

    def get_staleness_multiplier(self, current_version, update_version):
        """
        Calculates staleness scaling multiplier based on version delay.
        Multiplier = 1 / (1 + beta * staleness)
        """
        staleness = max(0, current_version - update_version)
        if staleness == 0:
            return 1.0
            
        # Inverse linear decay or exponential decay
        multiplier = 1.0 / (1.0 + self.beta * staleness)
        return float(multiplier)

    def process_asynchronous_update(self, base_eta, current_version, update_version):
        """Returns the scaled eta parameter accounting for staleness."""
        multiplier = self.get_staleness_multiplier(current_version, update_version)
        scaled_eta = base_eta * multiplier
        return scaled_eta
