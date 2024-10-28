import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt

from torchfuncbasis.smoother import points2basiscoefs
from torchfuncbasis.basis.bsplinebasis import BSplineBasis

def test_smoothing():
    # Set random seed for reproducibility
    # torch.manual_seed(42)

    # Define parameters
    batch_size = 5
    n_points = 100
    n_basis = 10
    n_channels = 2
    order = 4
    domain_range = (0., 1.)
    noise_level = 0.02

    # Create BSpline basis
    basis = BSplineBasis(
        n_basis=n_basis,
        domain_range=domain_range,
        order=order
    )

    # Generate random coefficients
    true_coefs = torch.randn(batch_size, n_channels, n_basis)

    # Generate evaluation points
    x = torch.linspace(domain_range[0], domain_range[1], n_points)
    x = x.unsqueeze(-1).expand(batch_size, n_points, 1)

    # Evaluate basis at points
    basis_mat = basis(x)  # (batch_size, n_points, n_basis)

    # Generate true values
    y_true = torch.einsum('...np,...cp->...nc', basis_mat, true_coefs)

    # Add noise
    noise = torch.randn_like(y_true) * noise_level
    y_noisy = y_true + noise

    # Recover coefficients using smoothing
    estimated_coefs = points2basiscoefs(
        x, y_noisy, basis,
        penalty_orders=2,
        smoothing_param=1e-7
    )

    # Plot comparison for three randomly selected curves
    selected_indices = torch.randperm(batch_size)[:3]
    fig, axes = plt.subplots(3, n_channels, figsize=(12, 12))

    for i, idx in enumerate(selected_indices):
        # Compute fitted values
        basis_mat_i = basis_mat[idx]
        y_fitted = torch.einsum('np,cp->nc', basis_mat_i, estimated_coefs[idx])

        for j in range(n_channels):
            ax = axes[i, j]
            ax.plot(x[idx, :, 0], y_true[idx, :, j], 'b-', label='True')
            ax.plot(x[idx, :, 0], y_noisy[idx, :, j],
                    'g.', label='Noisy', alpha=0.3)
            ax.plot(x[idx, :, 0], y_fitted[:, j], 'r--', label='Fitted')
            ax.set_title(f'Curve {idx}, Channel {j}')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Compute coefficient differences
    coef_diff = torch.norm(estimated_coefs - true_coefs, dim=-1)
    print("\nCoefficient differences (Frobenius norm per curve):")
    print(coef_diff)

    # Compute overall error
    total_error = torch.norm(estimated_coefs - true_coefs)
    print(f"\nTotal coefficient error (Frobenius norm): {total_error:.6f}")


if __name__ == "__main__":
    test_smoothing()
