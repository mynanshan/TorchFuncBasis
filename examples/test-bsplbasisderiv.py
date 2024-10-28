import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from torchfuncbasis.basis._bspline import B_spline, B_basis_deriv


def test_bspline_basis_derivatives():
    k = 3  # cubic spline
    nbreaks = 7
    deriv_order = 1  # Test first derivative

    # Generate knots
    t = torch.linspace(0., 1., nbreaks)
    # Pad the knots
    pad_left = torch.full((k,), t[0].item())
    pad_right = torch.full((k,), t[-1].item())
    t = torch.cat([pad_left, t, pad_right])

    # Generate evaluation points
    x = torch.linspace(0, 1, 100)

    # Compute basis derivatives using custom implementation
    basis_deriv_custom = B_basis_deriv(x, t, k=k, deriv=deriv_order)

    # Compute basis derivatives using SciPy
    # We need to compute each basis function separately in SciPy
    n_basis = len(t) - k - 1
    basis_deriv_scipy = np.zeros((len(x), n_basis))

    for i in range(n_basis):
        # Create coefficient vector that selects just one basis function
        c = np.zeros(n_basis)
        c[i] = 1.0
        tck = (t.numpy(), c, k)
        basis_deriv_scipy[:, i] = interpolate.splev(x.numpy(), tck, der=deriv_order)

    # Plot comparison for a few basis functions
    plt.figure(figsize=(15, 10))
    basis_to_plot = [1, n_basis//2, n_basis-2]  # Plot first, middle, and last basis

    for idx, i in enumerate(basis_to_plot):
        plt.subplot(2, len(basis_to_plot), idx+1)
        plt.plot(x.numpy(), basis_deriv_custom[:, i].numpy(), 'b-',
                label=f'Custom Basis {i}')
        plt.plot(x.numpy(), basis_deriv_scipy[:, i], 'r--',
                label=f'SciPy Basis {i}')
        plt.legend()
        plt.title(f'Basis {i} First Derivative')

        # Plot error
        plt.subplot(2, len(basis_to_plot), len(basis_to_plot)+idx+1)
        error = np.abs(basis_deriv_custom[:, i].numpy() - basis_deriv_scipy[:, i])
        plt.plot(x.numpy(), error, 'g-', label='Absolute Error')
        plt.legend()
        plt.title(f'Absolute Error for Basis {i}')

    plt.tight_layout()
    plt.show()

    # Compute maximum error across all basis functions
    max_error = np.max(np.abs(basis_deriv_custom.numpy() - basis_deriv_scipy))
    print(f"Maximum absolute error across all basis derivatives: {max_error}")

    # Test second derivative
    deriv_order = 2
    basis_deriv2_custom = B_basis_deriv(x, t, k=k, deriv=deriv_order)

    basis_deriv2_scipy = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        tck = (t.numpy(), c, k)
        basis_deriv2_scipy[:, i] = interpolate.splev(x.numpy(), tck, der=deriv_order)

    # Plot second derivatives
    plt.figure(figsize=(15, 10))
    for idx, i in enumerate(basis_to_plot):
        plt.subplot(2, len(basis_to_plot), idx+1)
        plt.plot(x.numpy(), basis_deriv2_custom[:, i].numpy(), 'b-',
                label=f'Custom Basis {i}')
        plt.plot(x.numpy(), basis_deriv2_scipy[:, i], 'r--',
                label=f'SciPy Basis {i}')
        plt.legend()
        plt.title(f'Basis {i} Second Derivative')

        plt.subplot(2, len(basis_to_plot), len(basis_to_plot)+idx+1)
        error = np.abs(basis_deriv2_custom[:, i].numpy() - basis_deriv2_scipy[:, i])
        plt.plot(x.numpy(), error, 'g-', label='Absolute Error')
        plt.legend()
        plt.title(f'Absolute Error for Basis {i}')

    plt.tight_layout()
    plt.show()

    max_error = np.max(np.abs(basis_deriv2_custom.numpy() - basis_deriv2_scipy))
    print(f"Maximum absolute error across all second derivatives: {max_error}")

if __name__ == "__main__":
    print("Testing B-spline basis derivatives...")
    test_bspline_basis_derivatives()
