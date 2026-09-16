"""
Percolation Simulation Module
=============================
Efficient simulation of standard (Erdős-Rényi) and explosive (Achlioptas Product Rule & Best-of-Two)
percolation phase transitions in complex networks using Union-Find (Disjoint-Set Data Structure).

References:
- Achlioptas, D., D'Souza, R. M., & Spencer, J. (2009). Explosive percolation in random networks. Science, 323(5920), 1453-1455.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


class DisjointSet:
    """Disjoint-set (Union-Find) data structure with path compression and size tracking."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.max_size = 1

    def find(self, i):
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            if self.size[root_i] > self.max_size:
                self.max_size = self.size[root_i]
            return True
        return False

    def get_size(self, i):
        return self.size[self.find(i)]

    def is_connected(self, i, j):
        return self.find(i) == self.find(j)


def simulate_er(n_nodes, max_edges=None):
    """Simulates Erdős-Rényi (ER) random percolation."""
    if max_edges is None:
        max_edges = int(1.6 * n_nodes)
        
    ds = DisjointSet(n_nodes)
    gcc_sizes = [ds.max_size / n_nodes]
    edge_densities = [0.0]
    
    for step in range(1, max_edges + 1):
        u, v = random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)
        ds.union(u, v)
        gcc_sizes.append(ds.max_size / n_nodes)
        edge_densities.append(step / n_nodes)
        
    return np.array(edge_densities), np.array(gcc_sizes)


def simulate_pr(n_nodes, max_edges=None):
    """Simulates Achlioptas Product Rule (PR) explosive percolation."""
    if max_edges is None:
        max_edges = int(1.6 * n_nodes)
        
    ds = DisjointSet(n_nodes)
    gcc_sizes = [ds.max_size / n_nodes]
    edge_densities = [0.0]
    
    for step in range(1, max_edges + 1):
        # Candidate 1
        u1, v1 = random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)
        w1 = ds.get_size(u1) * ds.get_size(v1) if not ds.is_connected(u1, v1) else float('inf')
        
        # Candidate 2
        u2, v2 = random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)
        w2 = ds.get_size(u2) * ds.get_size(v2) if not ds.is_connected(u2, v2) else float('inf')
        
        # Add edge minimizing product of component sizes
        if w1 <= w2:
            ds.union(u1, v1)
        else:
            ds.union(u2, v2)
            
        gcc_sizes.append(ds.max_size / n_nodes)
        edge_densities.append(step / n_nodes)
        
    return np.array(edge_densities), np.array(gcc_sizes)


def simulate_bf(n_nodes, max_edges=None):
    """Simulates Best-of-Two / Bounded-Size (BF) percolation rule."""
    if max_edges is None:
        max_edges = int(1.6 * n_nodes)
        
    ds = DisjointSet(n_nodes)
    gcc_sizes = [ds.max_size / n_nodes]
    edge_densities = [0.0]
    
    for step in range(1, max_edges + 1):
        u1, v1 = random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)
        u2, v2 = random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)
        
        prod1 = ds.get_size(u1) * ds.get_size(v1)
        if prod1 == 1:
            ds.union(u1, v1)
        else:
            ds.union(u2, v2)
            
        gcc_sizes.append(ds.max_size / n_nodes)
        edge_densities.append(step / n_nodes)
        
    return np.array(edge_densities), np.array(gcc_sizes)


def plot_percolation_comparison(n_nodes=20000, save_path="explosive_percolation_transition.png"):
    """Runs all three percolation models and plots the transition curves."""
    print(f"Running percolation simulations for N = {n_nodes:,} nodes...")
    
    t_er, gcc_er = simulate_er(n_nodes)
    t_pr, gcc_pr = simulate_pr(n_nodes)
    t_bf, gcc_bf = simulate_bf(n_nodes)
    
    plt.figure(figsize=(9, 6))
    plt.plot(t_er, gcc_er, label="Erdős-Rényi (ER) Continuous", color="#2ca02c", linewidth=2)
    plt.plot(t_pr, gcc_pr, label="Achlioptas Product Rule (PR) Explosive", color="#d62728", linewidth=2)
    plt.plot(t_bf, gcc_bf, label="Bounded-Size (BF) Rule", color="#1f77b4", linewidth=2)
    
    plt.axvline(0.5, color="#2ca02c", linestyle="--", alpha=0.6, label="ER Critical Point ($t_c=0.5$)")
    plt.axvline(0.888, color="#d62728", linestyle="--", alpha=0.6, label="PR Critical Point ($t_c\\approx 0.888$)")
    
    plt.xlabel("Edge Density ($r = M / N$)", fontsize=12)
    plt.ylabel("Giant Component Fraction ($S_{max} / N$)", fontsize=12)
    plt.title(f"Percolation Phase Transitions in Random Networks ($N = {n_nodes:,}$)", fontsize=14, fontweight="bold")
    plt.xlim(-0.02, 1.6)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=11, loc="upper left")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_percolation_comparison(n_nodes=20000)

