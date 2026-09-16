import torch

class SecureUpdateValidator:
    def __init__(self, max_norm_threshold=100.0):
        """
        max_norm_threshold: Maximum allowable L2 weight difference norm.
        Updates exceeding this threshold indicate potential model poisoning attacks.
        """
        self.max_norm_threshold = max_norm_threshold

    def validate_update(self, current_state_dict, proposed_state_dict):
        """
        Performs structural and values checks on a proposed state dict.
        Returns:
            is_valid (bool), error_msg (str or None)
        """
        total_l2_diff = 0.0
        
        # 1. Structural Checks: check if keys match
        current_keys = set(current_state_dict.keys())
        proposed_keys = set(proposed_state_dict.keys())
        
        if current_keys != proposed_keys:
            missing = current_keys - proposed_keys
            extra = proposed_keys - current_keys
            return False, f"Structural mismatch. Missing keys: {missing}, Extra keys: {extra}"

        # 2. Deep parameter validation
        for k in current_state_dict.keys():
            t_curr = current_state_dict[k]
            t_prop = proposed_state_dict[k]
            
            # A. Shape check
            if t_curr.shape != t_prop.shape:
                return False, f"Shape mismatch on key '{k}': expected {t_curr.shape}, got {t_prop.shape}"
                
            # B. Dtype check
            if t_curr.dtype != t_prop.dtype:
                return False, f"Data type mismatch on key '{k}': expected {t_curr.dtype}, got {t_prop.dtype}"
                
            # Perform value checks only on floating point values
            if t_prop.is_floating_point():
                # C. Check NaN / Inf presence
                if torch.isnan(t_prop).any():
                    return False, f"NaN values detected in key '{k}'"
                if torch.isinf(t_prop).any():
                    return False, f"Inf values detected in key '{k}'"
                    
                # D. Accumulate L2 weight difference
                diff_norm = torch.norm(t_prop.float() - t_curr.float()).item()
                total_l2_diff += diff_norm ** 2
                
        # 3. Model Poisoning Anomaly Check
        total_l2_diff = total_l2_diff ** 0.5
        if total_l2_diff > self.max_norm_threshold:
            return False, f"Poisoning anomaly check failed. Update L2 difference norm is {total_l2_diff:.2f} (max allowed: {self.max_norm_threshold:.2f})"

        return True, None
if __name__ == '__main__':
    # Test validation
    validator = SecureUpdateValidator(max_norm_threshold=10.0)
    curr = {'weight': torch.ones(2, 2), 'bias': torch.zeros(2)}
    
    # Valid proposal
    prop_valid = {'weight': torch.ones(2, 2) + 0.1, 'bias': torch.zeros(2) + 0.05}
    valid, msg = validator.validate_update(curr, prop_valid)
    print("Valid update checks passed:", valid, msg)
    
    # Poisoned proposal (large shift)
    prop_poisoned = {'weight': torch.ones(2, 2) * 50.0, 'bias': torch.zeros(2)}
    valid, msg = validator.validate_update(curr, prop_poisoned)
    print("Poisoned update rejected:", not valid, msg)
    
    # NaN proposal
    prop_nan = {'weight': torch.tensor([[1.0, float('nan')], [1.0, 1.0]]), 'bias': torch.zeros(2)}
    valid, msg = validator.validate_update(curr, prop_nan)
    print("NaN update rejected:", not valid, msg)
