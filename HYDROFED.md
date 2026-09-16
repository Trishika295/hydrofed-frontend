# HydroFed Decentralized Model Consensus

This document covers the mathematical formulations, physical analogies, and asynchronous protocols of the HydroFed algorithm.

---

## 1. The Water-Flow Analogy
In centralized federated learning (e.g. FedAvg), client nodes transmit parameters to a central server that averages them. In a decentralized layout (e.g., Gossip, HydroFed), clinic nodes communicate only with direct neighbors in a sparse network graph.

HydroFed maps model weights to connected water containers:
- **Model discrepancy / weight distance** maps to **hydraulic pressure**.
- **Network bandwidth / communication quality** maps to **flow resistance**.
- **Consensus adjustments** map to **symmetric water flows** that balance containers.

---

## 2. Mathematical Formulations

### A. Pressure-Driven Flow Coefficient
For neighboring clinics $i$ and $j$, the weight discrepancy is defined as:
$$D_{ij} = \|W_i - W_j\|_2$$
Where $W$ represents the weights of target classifier and cross-attention projection layers.
The adaptive flow coefficient $\eta_{ij}$ scales dynamically based on this discrepancy and link resistance $R_{ij}$:
$$\eta_{ij} = \text{clip}\left(\gamma \cdot \frac{D_{ij}}{R_{ij}}, \, 0, \, \eta_{\text{max}}\right)$$
Where:
- $\gamma$ is the flow scaling coefficient (default 0.1).
- $\eta_{\text{max}}$ is the stability bound limit (default 0.45), preventing model weight updates from overshooting.

### B. Update-Conservation Principle
To preserve model updates before local training iterations, updates between neighbors are symmetric:
$$\Delta = \eta_{ij} (W_j - W_i)$$
$$W_i' = W_i + \Delta$$
$$W_j' = W_j - \Delta$$
This prevents parameter leakage/drift and mimics the physical conservation of water volumes in connected tanks.

### C. Asynchronous Staleness Mitigation
In real-world networks, updates arrive asynchronously. Stale updates from slower nodes are scaled down based on version delay:
$$S = V_{\text{current}} - V_{\text{update}}$$
$$\eta_{\text{stale}} = \eta \cdot \frac{1}{1 + \beta \cdot S}$$
Where $\beta$ is the decay scale (default 0.2). If $\beta=0.2$ and lag is 5 rounds, $\eta$ is scaled by $0.5\times$, shielding the network from stale gradients.
