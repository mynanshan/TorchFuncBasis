import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import matplotlib.pyplot as plt
from torchfuncbasis.basis.bsplinebasis import BSplineBasis
import numpy as np
from scipy import interpolate

def main():
    # Define the domain range and number of basis functions
    domain_range = (0.0, 1.0)
    n_basis = 5
    order = 4

    # Create a BSplineBasis object
    bspline_basis = BSplineBasis(
        n_basis=n_basis,
        domain_range=domain_range,
        order=order
    )

    # Define evaluation points
    eval_points = torch.linspace(
        domain_range[0], domain_range[1], 100)

    # Evaluate the B-spline basis at the evaluation points
    basis_matrix = bspline_basis(eval_points)
    basis_derivs = bspline_basis(eval_points, derivative=1)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the B-spline basis functions
    for i in range(n_basis):
        ax1.plot(eval_points.squeeze().numpy(),
                basis_matrix[:, i].numpy(), label=f'B-spline {i+1}')
    ax1.set_title('B-spline Basis Functions')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    # Plot the B-spline basis derivatives
    for i in range(n_basis):
        ax2.plot(eval_points.squeeze().numpy(),
                basis_derivs[:, i].numpy(), label=f'B-spline {i+1}')
    ax2.set_title('B-spline Basis Function Derivatives')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Value')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Get Gram matrix from custom implementation
    gram = bspline_basis.gram_matrix()
    print(gram.numpy())

if __name__ == "__main__":
    main()
