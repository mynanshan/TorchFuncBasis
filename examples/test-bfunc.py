import sys
import os
import numpy as np
from scipy.interpolate import BSpline

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import matplotlib.pyplot as plt
from torchfuncbasis.basis._bspline import B_basis

torch.manual_seed(2501)

def test_bspline_matrix():
    # Set up parameters
    num_splines = 2
    num_samples = 1001
    num_grid_points = 11
    k = 3  # spline order

    # Generate input x
    x = torch.linspace(0., 1, num_samples).unsqueeze(0).repeat(num_splines, 1)

    # Generate grid
    # grid = torch.linspace(0, 1, num_grid_points)
    grid = torch.rand((num_splines, num_grid_points - 2))
    grid, _ = torch.sort(grid, dim = -1)
    # expand the grid
    tol = 1e-6
    grid = torch.hstack([
        torch.full((num_splines, k+1), 0.),
        grid,
        torch.full((num_splines, k+1), 1.)
    ])
    print(grid[0].tolist())

    # Compute B-spline basis
    bspline_basis = B_basis(x, grid, k=k)

    # Compute spline basis with SciPy
    spline_list = []

    for i in range(bspline_basis.shape[2]):
        coef = np.zeros(bspline_basis.shape[2])
        coef[i] = 1.0
        spline = BSpline(grid[0].numpy(), coef, k)
        spline_list.append(spline(x[0].numpy()))

    scipy_basis = np.stack(spline_list, axis=1)

    diff = np.abs(bspline_basis[0].numpy() - scipy_basis)
    print(f"Maximum absolute difference: {np.max(diff)}")

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Plot the B-spline basis from your implementation
    for i in range(bspline_basis.shape[2]):
        axs[0].plot(x[0].numpy(), bspline_basis[0, :, i].numpy(), label=f'Basis {i+1}')

    axs[0].set_title(f'B-spline Basis Functions (order {k}) - Custom Implementation')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Basis Function Value')
    axs[0].legend()
    axs[0].grid(True)

    # Plot the B-spline basis from SciPy
    for i in range(scipy_basis.shape[1]):
        axs[1].plot(x[0].numpy(), scipy_basis[:, i], label=f'Basis {i+1}')

    axs[1].set_title(f'B-spline Basis Functions (order {k}) - SciPy')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Basis Function Value')
    axs[1].legend()
    axs[1].grid(True)

    # Plot the difference between the two bases
    for i in range(scipy_basis.shape[1]):
        axs[2].plot(x[0].numpy(), diff[:, i], label=f'Difference {i+1}')

    axs[2].set_title('Difference Between Custom and SciPy B-spline Basis')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Absolute Difference')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print some information
    print(f"Input shape: {x.shape}")
    print(f"Grid shape: {grid.shape}")
    print(f"B-spline basis shape: {bspline_basis.shape}")

if __name__ == "__main__":
    test_bspline_matrix()
