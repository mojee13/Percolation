"""
Legacy Test Script for Percolation Simulation
==============================================
Runs comparative simulations for Erdős-Rényi (ER), Achlioptas Product Rule (PR),
and Bounded-Size (BF) percolation models.
"""

from percolation_simulation import plot_percolation_comparison

if __name__ == "__main__":
    plot_percolation_comparison(n_nodes=16000, save_path="explosive_percolation_transition.png")
