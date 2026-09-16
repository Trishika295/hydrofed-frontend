import unittest
import torch
from cryptography.exceptions import InvalidTag
from security.aes_gcm import AESGCMEncryptor
from security.key_manager import PairwiseKeyManager
from security.authentication import ClinicAuthenticationService
from security.secure_update import SecureUpdateValidator

class TestSecurityServices(unittest.TestCase):
    def test_aes_gcm_roundtrip(self):
        encryptor = AESGCMEncryptor()
        state_dict = {
            'weight_layer': torch.randn(3, 3),
            'bias_layer': torch.tensor([0.1, 0.2, 0.3])
        }
        metadata = "round=5;sender=Client-12"
        
        # Encrypt
        nonce, ciphertext = encryptor.encrypt_state_dict(state_dict, metadata)
        self.assertEqual(len(nonce), 12)
        
        # Decrypt with correct key and metadata
        decrypted = encryptor.decrypt_state_dict(nonce, ciphertext, metadata)
        self.assertTrue(torch.equal(decrypted['bias_layer'], state_dict['bias_layer']))

    def test_aes_gcm_tampering_detection(self):
        encryptor = AESGCMEncryptor()
        state_dict = {'weight': torch.randn(2, 2)}
        metadata = "auth"
        
        nonce, ciphertext = encryptor.encrypt_state_dict(state_dict, metadata)
        
        # 1. Test incorrect metadata (associated data mismatch)
        with self.assertRaises(InvalidTag):
            encryptor.decrypt_state_dict(nonce, ciphertext, "wrong_auth")
            
        # 2. Test corrupted ciphertext (modify 1 byte)
        corrupted = bytearray(ciphertext)
        corrupted[4] ^= 0x55
        corrupted = bytes(corrupted)
        with self.assertRaises(InvalidTag):
            encryptor.decrypt_state_dict(nonce, corrupted, metadata)

    def test_pairwise_key_generation(self):
        km = PairwiseKeyManager("test_secret")
        key_ab = km.get_pairwise_key(0, 1)
        key_ba = km.get_pairwise_key(1, 0)
        
        # Symmetry check
        self.assertEqual(key_ab, key_ba)
        self.assertEqual(len(key_ab), 32)  # 256-bit key

    def test_clinic_authentication(self):
        auth = ClinicAuthenticationService([0, 1, 2], shared_secret="secret")
        token_0 = auth.get_auth_token(0)
        token_1 = auth.get_auth_token(1)
        
        self.assertTrue(auth.authenticate_sender(0, token_0))
        self.assertFalse(auth.authenticate_sender(0, token_1))  # wrong token
        self.assertFalse(auth.authenticate_sender(99, token_0)) # invalid clinic

    def test_secure_update_validation(self):
        validator = SecureUpdateValidator(max_norm_threshold=10.0)
        curr = {'weight': torch.ones(2, 2), 'bias': torch.zeros(2)}
        
        # 1. Mismatched shapes
        bad_shape = {'weight': torch.ones(2, 3), 'bias': torch.zeros(2)}
        valid, err = validator.validate_update(curr, bad_shape)
        self.assertFalse(valid)
        self.assertIn("Shape mismatch", err)
        
        # 2. NaN values
        bad_nan = {'weight': torch.tensor([[1.0, float('nan')], [1.0, 1.0]]), 'bias': torch.zeros(2)}
        valid, err = validator.validate_update(curr, bad_nan)
        self.assertFalse(valid)
        self.assertIn("NaN values detected", err)
        
        # 3. Anomaly poisoning (large L2 norm shift)
        bad_norm = {'weight': torch.ones(2, 2) * 20.0, 'bias': torch.zeros(2)}
        valid, err = validator.validate_update(curr, bad_norm)
        self.assertFalse(valid)
        self.assertIn("Poisoning anomaly", err)

if __name__ == '__main__':
    unittest.main()
