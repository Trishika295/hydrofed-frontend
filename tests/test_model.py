import unittest
import torch
from models.multimodal_model import HydroFedMultimodalModel
from models.frequency_branch import FrequencyFeatureBranch
from models.clinical_encoder import ClinicalEncoder
from models.cross_attention import MultimodalCrossAttention
from models.immune_feature_response import ImmuneFeatureResponse
from models.bmtf import BioMultiscaleThreatFusion
from models.adaptive_immune_gate import AdaptiveImmuneCrossAttentionGate
from models.uncertainty import estimate_mc_uncertainty

class TestMultimodalArchitecture(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 128
        self.batch_size = 2
        
    def test_frequency_branch(self):
        branch = FrequencyFeatureBranch(embed_dim=self.embed_dim)
        x = torch.randn(self.batch_size, 3, 224, 224)
        out = branch(x)
        self.assertEqual(out.shape, (self.batch_size, self.embed_dim, 7, 7))

    def test_clinical_encoder(self):
        encoder = ClinicalEncoder(embed_dim=self.embed_dim)
        # clinical features: age_norm, gender, diabetes, smoke, family
        x = torch.tensor([[0.2, 0, 1, 0, 1], [0.8, 1, 0, 1, 0]], dtype=torch.float32)
        out = encoder(x)
        self.assertEqual(out.shape, (self.batch_size, 5, self.embed_dim))

    def test_iifr_amplification(self):
        iifr = ImmuneFeatureResponse(embed_dim=self.embed_dim)
        x = torch.randn(self.batch_size, self.embed_dim, 7, 7)
        out = iifr(x)
        self.assertEqual(out.shape, x.shape)

    def test_bmtf_fusion(self):
        bmtf = BioMultiscaleThreatFusion(embed_dim=self.embed_dim)
        fi = torch.randn(self.batch_size, self.embed_dim, 7, 7)
        ff = torch.randn(self.batch_size, self.embed_dim, 7, 7)
        out = bmtf(fi, ff)
        self.assertEqual(out.shape, fi.shape)

    def test_cross_attention(self):
        attn = MultimodalCrossAttention(embed_dim=self.embed_dim)
        xray = torch.randn(self.batch_size, self.embed_dim, 7, 7)
        clinical = torch.randn(self.batch_size, 5, self.embed_dim)
        out = attn(xray, clinical)
        self.assertEqual(out.shape, xray.shape)
        self.assertIsNotNone(attn.attention_weights)

    def test_aica_gate(self):
        gate = AdaptiveImmuneCrossAttentionGate(embed_dim=self.embed_dim)
        xray = torch.randn(self.batch_size, self.embed_dim, 7, 7)
        ca = torch.randn(self.batch_size, self.embed_dim, 7, 7)
        out = gate(xray, ca)
        self.assertEqual(out.shape, xray.shape)

    def test_full_model_forward(self):
        model = HydroFedMultimodalModel(embed_dim=self.embed_dim, num_classes=2, pretrained_backbone=False)
        xray = torch.randn(self.batch_size, 3, 224, 224)
        clinical = torch.tensor([[0.2, 0, 1, 0, 1], [0.8, 1, 0, 1, 0]], dtype=torch.float32)
        logits = model(xray, clinical)
        self.assertEqual(logits.shape, (self.batch_size, 2))

    def test_uncertainty_estimation(self):
        model = HydroFedMultimodalModel(embed_dim=self.embed_dim, num_classes=2, pretrained_backbone=False)
        xray = torch.randn(1, 3, 224, 224)
        clinical = torch.tensor([[0.2, 0, 1, 0, 1]], dtype=torch.float32)
        res = estimate_mc_uncertainty(model, xray, clinical, num_passes=5)
        
        self.assertIn('mean_prob', res)
        self.assertIn('confidence', res)
        self.assertIn('uncertainty', res)
        self.assertIn('predicted_class', res)
        self.assertEqual(len(res['raw_probs']), 5)

if __name__ == '__main__':
    unittest.main()
