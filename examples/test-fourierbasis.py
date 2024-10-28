import numpy as np
from torchfuncbasis.basis.fourierbasis import FourierBasis
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main():
    # Define the domain range, number of basis functions, and period
    domain_range = (0.0, 2 * np.pi)
    n_basis = 5
    period = 2 * np.pi

    # Create a FourierBasis object
    fourier_basis = FourierBasis(
        n_basis=n_basis,
        domain_range=domain_range,
        period=period
    )

    # Define evaluation points
    eval_points = torch.linspace(
        domain_range[0], domain_range[1], 200).unsqueeze(-1)

    # Evaluate the Fourier basis at the evaluation points
    basis_matrix = fourier_basis(eval_points)
    basis_derivs = fourier_basis(eval_points, derivative=1)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the Fourier basis functions
    for i in range(n_basis):
        ax1.plot(eval_points.squeeze().numpy(),
                 basis_matrix[:, i].numpy(), label=f'Fourier {i+1}')
    ax1.set_title('Fourier Basis Functions')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)

    # Plot the Fourier basis derivatives
    for i in range(n_basis):
        ax2.plot(eval_points.squeeze().numpy(),
                 basis_derivs[:, i].numpy(), label=f'Fourier {i+1}')
    ax2.set_title('Fourier Basis Function Derivatives')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print some information about the basis
    print(f"Number of basis functions: {fourier_basis.n_basis}")
    print(f"Domain range: {fourier_basis.domain_range}")
    print(f"Period: {fourier_basis.period}")

    # Test with batch dimensions
    batch_eval_points = torch.rand(
        3, 4, 100, 1) * (domain_range[1] - domain_range[0]) + domain_range[0]
    batch_result = fourier_basis(batch_eval_points)
    print(f"Shape of batch evaluation result: {batch_result.shape}")

    print(fourier_basis.gram_matrix().numpy())
    print(fourier_basis.gram_matrix(derivative=2).numpy())


if __name__ == "__main__":
    main()
