import unittest
import torch
from federated.topology import generate_small_world_topology
from federated.hydrofed import HydroFedConsensusAggregator
from federated.clinic import LocalClinicNode
from federated.asynchronous import AsynchronousHydroFedController

class TestFederatedEngine(unittest.TestCase):
    def test_small_world_topology(self):
        num_nodes = 60
        adj = generate_small_world_topology(num_nodes, k=4, p=0.15)
        
        # Verify node counts and graph connectivity
        self.assertEqual(len(adj), num_nodes)
        
        # Verify undirected graph symmetry
        for node, neighbors in adj.items():
            for neighbor in neighbors:
                self.assertIn(node, adj[neighbor])
                
    def test_hydrofed_weight_conservation(self):
        aggregator = HydroFedConsensusAggregator(gamma=0.5, eta_max=0.5)
        
        # Set up two mock clinic nodes with 2 parameters
        node_a = LocalClinicNode(client_id=0, registry_csv_path="reports/final_metadata_registry.csv", embed_dim=8)
        node_b = LocalClinicNode(client_id=1, registry_csv_path="reports/final_metadata_registry.csv", embed_dim=8)
        
        # Modify weights to create differences
        dict_a = node_a.get_model_parameters()
        dict_b = node_b.get_model_parameters()
        
        # Force a simple coordinate update
        dict_a['classifier.weight'].fill_(1.0)
        dict_b['classifier.weight'].fill_(5.0)
        
        # Store sum prior to exchange
        sum_pre = dict_a['classifier.weight'].sum().item() + dict_b['classifier.weight'].sum().item()
        
        node_a.load_model_parameters(dict_a)
        node_b.load_model_parameters(dict_b)
        
        # Run consensus step (eta is computed from L2 difference)
        disagreement, eta = aggregator.execute_pairwise_consensus(node_a, node_b)
        
        # Extract states post exchange
        new_dict_a = node_a.get_model_parameters()
        new_dict_b = node_b.get_model_parameters()
        
        sum_post = new_dict_a['classifier.weight'].sum().item() + new_dict_b['classifier.weight'].sum().item()
        
        # Weight Conservation Check: sum_pre must equal sum_post
        self.assertAlmostEqual(sum_pre, sum_post, places=4)
        
        # Model Convergence check: the parameters must have moved closer
        diff_pre = torch.norm(dict_a['classifier.weight'] - dict_b['classifier.weight']).item()
        diff_post = torch.norm(new_dict_a['classifier.weight'] - new_dict_b['classifier.weight']).item()
        self.assertTrue(diff_post < diff_pre)

    def test_async_staleness_decay(self):
        async_ctrl = AsynchronousHydroFedController(beta=0.2)
        base_eta = 0.4
        
        # 1. Update is fresh (no delay)
        eta_fresh = async_ctrl.process_asynchronous_update(base_eta, current_version=5, update_version=5)
        self.assertEqual(eta_fresh, base_eta)
        
        # 2. Update is stale (lag = 5 versions)
        # Expected: 0.4 * 1 / (1 + 0.2 * 5) = 0.4 * 0.5 = 0.2
        eta_stale = async_ctrl.process_asynchronous_update(base_eta, current_version=10, update_version=5)
        self.assertAlmostEqual(eta_stale, 0.2, places=4)
        
        # 3. Update is extremely stale
        eta_very_stale = async_ctrl.process_asynchronous_update(base_eta, current_version=50, update_version=5)
        self.assertTrue(eta_very_stale < eta_stale)

if __name__ == '__main__':
    unittest.main()
