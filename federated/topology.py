import random
import numpy as np

def generate_ring_topology(num_nodes=60):
    """Ring layout: each node connected to immediate left and right neighbors."""
    adj = {i: set() for i in range(num_nodes)}
    for i in range(num_nodes):
        left = (i - 1) % num_nodes
        right = (i + 1) % num_nodes
        adj[i].add(left)
        adj[i].add(right)
        adj[left].add(i)
        adj[right].add(i)
    return {k: list(v) for k, v in adj.items()}

def generate_grid_topology(num_nodes=60):
    """Grid layout: nearest square grid dimensions (e.g., 6x10 or 8x8)."""
    # 60 nodes fits neatly in a 6x10 grid
    rows = 6
    cols = 10
    assert rows * cols == num_nodes, "num_nodes must equal rows * cols"
    
    adj = {i: set() for i in range(num_nodes)}
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            # Connect to right
            if c < cols - 1:
                right = r * cols + (c + 1)
                adj[node].add(right)
                adj[right].add(node)
            # Connect to bottom
            if r < rows - 1:
                bottom = (r + 1) * cols + c
                adj[node].add(bottom)
                adj[bottom].add(node)
    return {k: list(v) for k, v in adj.items()}

def generate_small_world_topology(num_nodes=60, k=4, p=0.15, seed=42):
    """
    Watts-Strogatz Small-World graph:
    1. Start with a ring of k nearest neighbors
    2. Rewire edges with probability p
    """
    random.seed(seed)
    np.random.seed(seed)
    
    adj = {i: set() for i in range(num_nodes)}
    
    # 1. Create a ring lattice where each node is connected to k nearest neighbors (k/2 on each side)
    for i in range(num_nodes):
        for neighbor_offset in range(1, k // 2 + 1):
            left = (i - neighbor_offset) % num_nodes
            right = (i + neighbor_offset) % num_nodes
            adj[i].add(left)
            adj[i].add(right)
            adj[left].add(i)
            adj[right].add(i)
            
    # 2. Rewire edges with probability p
    # For each node, rewrites its right-side edges
    for i in range(num_nodes):
        for neighbor_offset in range(1, k // 2 + 1):
            right = (i + neighbor_offset) % num_nodes
            if random.random() < p:
                # Rewire right edge
                # Find a new target node that is not i and not already connected to i
                potential_targets = [n for n in range(num_nodes) if n != i and n not in adj[i]]
                if potential_targets:
                    new_target = random.choice(potential_targets)
                    # Disconnect old right link
                    if right in adj[i]: adj[i].remove(right)
                    if i in adj[right]: adj[right].remove(i)
                    
                    # Connect new rewired link
                    adj[i].add(new_target)
                    adj[new_target].add(i)
                    
    # Guarantee connectivity (if graph breaks, fallback to pure ring)
    # Simple BFS check
    visited = set()
    queue = [0]
    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            visited.add(curr)
            queue.extend([n for n in adj[curr] if n not in visited])
            
    if len(visited) < num_nodes:
        # Connected graph check failed, falling back to clean ring lattice layout
        print("Warning: Small-world generated disconnected layout. Falling back to ring lattice.")
        return generate_ring_topology(num_nodes)

    return {k: list(v) for k, v in adj.items()}

if __name__ == '__main__':
    adj = generate_small_world_topology(60)
    print("Clinic 0 neighbors:", adj[0])
    print("Clinic 20 neighbors:", adj[20])
    total_edges = sum(len(v) for v in adj.values()) // 2
    print("Total undirected communication edges in graph:", total_edges)
