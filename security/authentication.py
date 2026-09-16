import hashlib

class ClinicAuthenticationService:
    def __init__(self, allowed_clinic_ids, shared_secret='handshake_token_secret_2026'):
        self.shared_secret = shared_secret
        # Pre-generate secure auth tokens for all allowed clinics
        self.auth_registry = {}
        for cid in allowed_clinic_ids:
            token = self._generate_token(str(cid))
            self.auth_registry[str(cid)] = token

    def _generate_token(self, clinic_id):
        """Generates a challenge token based on clinic ID and shared secret."""
        h = hashlib.sha256()
        h.update(clinic_id.encode('utf-8'))
        h.update(self.shared_secret.encode('utf-8'))
        return h.hexdigest()

    def get_auth_token(self, clinic_id):
        """Allows a clinic node to fetch its own authentication token."""
        return self.auth_registry.get(str(clinic_id))

    def authenticate_sender(self, sender_id, submitted_token):
        """
        Authenticates an incoming message from sender_id.
        Returns:
            bool: True if authenticated, False otherwise
        """
        sender_id_str = str(sender_id)
        if sender_id_str not in self.auth_registry:
            return False
            
        expected_token = self.auth_registry[sender_id_str]
        # Constant-time comparison to protect against timing attacks
        return hmac_compare(expected_token, submitted_token)

def hmac_compare(val1, val2):
    """Simple constant-time comparison for tokens to prevent timing analysis."""
    if len(val1) != len(val2):
        return False
    result = 0
    for x, y in zip(val1, val2):
        result |= ord(x) ^ ord(y)
    return result == 0
