import os
import pickle
import io
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

class AESGCMEncryptor:
    def __init__(self, key=None):
        """
        Initializes the AES-256-GCM encryptor.
        If no key is provided, generates a cryptographically secure 256-bit key.
        """
        if key is None:
            self.key = AESGCM.generate_key(bit_length=256)
        else:
            self.key = key
        self.aesgcm = AESGCM(self.key)

    def encrypt_state_dict(self, state_dict, associated_data=None):
        """
        Serializes and encrypts a PyTorch state_dict.
        associated_data: optional metadata (e.g., model_version, client_id) for verification
        Returns:
            nonce: 12-byte secure random nonce
            ciphertext: encrypted bytes (includes the 16-byte authentication tag)
        """
        # 1. Serialize model state dict using pickle to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        serialized = buffer.getvalue()
        
        # 2. Generate 12-byte secure random nonce
        nonce = os.urandom(12)
        
        # 3. Authenticated encryption
        aad_bytes = associated_data.encode('utf-8') if associated_data else None
        ciphertext = self.aesgcm.encrypt(nonce, serialized, aad_bytes)
        
        return nonce, ciphertext

    def decrypt_state_dict(self, nonce, ciphertext, associated_data=None):
        """
        Decrypts and de-serializes a PyTorch state_dict.
        Raises InvalidTag if ciphertext is tampered or key is incorrect.
        """
        aad_bytes = associated_data.encode('utf-8') if associated_data else None
        
        # Authenticated decryption (raises InvalidTag on failure)
        decrypted = self.aesgcm.decrypt(nonce, ciphertext, aad_bytes)
        
        # De-serialize state dict
        buffer = io.BytesIO(decrypted)
        state_dict = torch.load(buffer)
        
        return state_dict

if __name__ == '__main__':
    # Test encryption roundtrip
    encryptor = AESGCMEncryptor()
    dummy_dict = {'weight': torch.randn(2, 2), 'bias': torch.tensor([0.1, 0.2])}
    
    metadata = "model_version=v1.0;client_id=Client-01"
    nonce, ciphertext = encryptor.encrypt_state_dict(dummy_dict, metadata)
    print("Encryption Succeeded. Ciphertext size:", len(ciphertext))
    
    # Decrypt with correct key
    decrypted = encryptor.decrypt_state_dict(nonce, ciphertext, metadata)
    print("Decryption Succeeded. Reconstructed bias:", decrypted['bias'])
    
    # Test decryption with tampered ciphertext
    tampered_ciphertext = bytearray(ciphertext)
    tampered_ciphertext[0] ^= 0x01  # flip 1 bit
    tampered_ciphertext = bytes(tampered_ciphertext)
    
    try:
        encryptor.decrypt_state_dict(nonce, tampered_ciphertext, metadata)
        print("FAIL: Tampered ciphertext was decrypted!")
    except InvalidTag:
        print("PASS: Decryption failed safely on tampered ciphertext (Invalid Tag detected).")
