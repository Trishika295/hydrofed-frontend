import torch

class DecentralizedGossip:
    def __init__(self):
        pass

    def execute_gossip_averaging(self, client_nodes, adjacency_list):
        """
        Executes standard gossip averaging:
        W_i_new = (W_i + sum_j W_j) / (d_i + 1)
        """
        # Store original parameters for all nodes before updating
        original_states = {node.client_id: node.get_model_parameters() for node in client_nodes}
        num_nodes = len(client_nodes)
        
        # Dictionary to map client_id to node object for easy lookup
        id_to_node = {node.client_id: node for node in client_nodes}
        
        for node in client_nodes:
            cid = node.client_id
            neighbors = adjacency_list.get(cid, [])
            active_neighbors = [n for n in neighbors if n in id_to_node]
            
            if not active_neighbors:
                continue
                
            divisor = len(active_neighbors) + 1
            new_state = {}
            ref_state = original_states[cid]
            
            for k in ref_state.keys():
                if ref_state[k].is_floating_point():
                    # Sum of local + neighbors
                    sum_weights = ref_state[k].clone()
                    for n_id in active_neighbors:
                        sum_weights += original_states[n_id][k]
                    new_state[k] = sum_weights / divisor
                else:
                    new_state[k] = ref_state[k].clone()
                    
            node.load_model_parameters(new_state)
            
        return
