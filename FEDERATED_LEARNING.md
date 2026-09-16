# Decentralized Federated Learning Architecture

This document covers the 60-clinic simulation layouts, partitioning parameters, and topology designs.

---

## 1. Network Topology Configs
We simulate 60 clinical nodes communicating over a decentralized graph topology. Adjacency maps support multiple structures:
- **Ring**: Simple ring lattice.
- **Grid**: 6x10 coordinate grid.
- **Connected Sparse Small-World (Default)**: Watts-Strogatz layout with $k=4$ nearest neighbors and shortcut rewiring probability $p=0.15$. This replicates cluster-based regional clinical networks connected by sparse long-range inter-regional pipelines.

---

## 2. Non-IID Class Partitioning
Prevalence of pneumonia classes varies between clinic demographics.
Using a Dirichlet distribution with parameter $\alpha = 0.5$ on class labels:
- Allocates training samples across the 60 nodes.
- Simulates realistic non-IID conditions where certain clinics observe mostly NORMAL cases, some mostly BACTERIAL, and some mostly VIRAL.

---

## 3. Gossip vs. Centralized Architectures

### A. Centralized Baselines (FedAvg / FedProx)
- Clients train locally on their private dataloader.
- Send model weights to a central server every round.
- Server averages parameters (weighted by client sample size) and broadcasts the aggregated weights back to all clients.
- **FedProx** adds a proximal regularization penalty ($\frac{\mu}{2} \|W - W_{\text{global}}\|_2^2$) to stabilize updates in strongly non-IID conditions.

### B. Decentralized Baselines (Gossip)
- No central server exists.
- At the end of each round, clients average their state dicts uniformly with all connected neighbors:
  $$W_i^{(t+1)} = \frac{1}{|\mathcal{N}(i)| + 1} \left(W_i^{(t)} + \sum_{j \in \mathcal{N}(i)} W_j^{(t)}\right)$$

### C. HydroFed Gossip Consensus (Our Method)
- Dynamically controls parameter flow between neighbors using model pressure (disagreement) and link resistance.
- Enforces weight conservation symmetric adjustments during communication rounds.
