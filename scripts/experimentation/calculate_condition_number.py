#!/usr/bin/env python3
"""
Calculate condition number of correlation matrix
"""

import numpy as np

# Correlation matrix from DCM 2-ROI MCMC results
corr_matrix = np.array(
    [
        [1.0, -0.7163871236107592, 0.5956378250091612, -0.9597702640741499],
        [-0.7163871236107592, 1.0, -0.9041824889901707, 0.6944912988815273],
        [0.5956378250091612, -0.9041824889901707, 1.0, -0.6686784146338731],
        [-0.9597702640741499, 0.6944912988815273, -0.6686784146338731, 1.0],
    ]
)

# Calculate eigenvalues
eigenvalues = np.linalg.eigvalsh(corr_matrix)

# Calculate condition number (ratio of max to min eigenvalue)
cond_number = np.max(eigenvalues) / np.min(eigenvalues)

# Calculate max off-diagonal correlation
off_diag_mask = ~np.eye(4, dtype=bool)
max_off_diag = np.max(np.abs(corr_matrix[off_diag_mask]))

print("Correlation Matrix:")
print(corr_matrix)
print("\nEigenvalues:", eigenvalues)
print(f"\nMax eigenvalue: {np.max(eigenvalues):.4f}")
print(f"Min eigenvalue: {np.min(eigenvalues):.4f}")
print(f"\nCondition number: {cond_number:.2f}")
print(f"Condition number (scientific): {cond_number:.2e}")
print(f"\nMax |corr_ij| (off-diagonal): {max_off_diag:.3f}")
