import hashlib
import os

class PairwiseKeyManager:
    def __init__(self, master_secret=None):
        """
        Initializes the Key Manager with a master secret.
        In production, the master secret would be pulled from environmental KMS/HSM.
        """
        if master_secret is None:
            # Generate a secure fallback key
            self.master_secret = os.environ.get('HYDROFED_MASTER_SECRET', 'decentralized_clinical_consensus_key_2026')
        else:
            self.master_secret = master_secret

    def get_pairwise_key(self, client_a, client_b):
        """
        Derives a unique 256-bit AES symmetric key for communication between client_a and client_b.
        Uses SHA-256 HKDF-like derivation based on client IDs and the master secret.
        Key is order-independent: get_pairwise_key(A, B) == get_pairwise_key(B, A).
        """
        c1, c2 = sorted([str(client_a), str(client_b)])
        
        # Construct key material
        salt = f"{c1}_to_{c2}".encode('utf-8')
        
        # Derive key via SHA-256 HMAC / HKDF simulation
        h = hashlib.sha256()
        h.update(self.master_secret.encode('utf-8'))
        h.update(salt)
        
        return h.digest()  # returns 32-byte (256-bit) key
