import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from torchfuncbasis.basis._bspline import B_spline

def test_bspline_derivatives():
    k = 3
    nbreaks = 9
    deriv_order = 1  # Test first derivative

    # Generate knots
    t = torch.linspace(0., 1., nbreaks)
    # Pad the knots
    pad_left = torch.full((k,), t[0].item())
    pad_right = torch.full((k,), t[-1].item())
    t = torch.cat([pad_left, t, pad_right])

    # Generate random coefficients
    n_coef = nbreaks + k - 1
    c = torch.rand(n_coef)

    # Generate evaluation points
    x = torch.linspace(0, 1, 100)

    # Compute derivative using custom implementation
    y_deriv_custom = B_spline(x, c, t, k=k, deriv=deriv_order)

    # Compute derivative using SciPy
    tck = (t.numpy(), c.numpy(), k)
    y_deriv_scipy = interpolate.splev(x.numpy(), tck, der=deriv_order)

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(x.numpy(), y_deriv_custom.numpy(), 'b-', label='Custom')
    plt.plot(x.numpy(), y_deriv_scipy, 'r--', label='SciPy')
    plt.legend()
    plt.title(f'B-Spline {deriv_order}st Derivative Comparison')

    # Plot error
    plt.subplot(2, 1, 2)
    error = np.abs(y_deriv_custom.numpy() - y_deriv_scipy)
    plt.plot(x.numpy(), error, 'g-', label='Absolute Error')
    plt.legend()
    plt.title('Absolute Error')
    plt.tight_layout()
    plt.show()

    # Compute maximum error
    max_error = np.max(np.abs(y_deriv_custom.numpy() - y_deriv_scipy))
    print(f"Maximum absolute error (derivative): {max_error}")

    # Test second derivative
    deriv_order = 2
    y_deriv2_custom = B_spline(x, c, t, k=k, deriv=deriv_order)
    y_deriv2_scipy = interpolate.splev(x.numpy(), tck, der=deriv_order)

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(x.numpy(), y_deriv2_custom.numpy(), 'b-', label='Custom')
    plt.plot(x.numpy(), y_deriv2_scipy, 'r--', label='SciPy')
    plt.legend()
    plt.title(f'B-Spline {deriv_order}nd Derivative Comparison')

    # Plot error
    plt.subplot(2, 1, 2)
    error = np.abs(y_deriv2_custom.numpy() - y_deriv2_scipy)
    plt.plot(x.numpy(), error, 'g-', label='Absolute Error')
    plt.legend()
    plt.title('Absolute Error')
    plt.tight_layout()
    plt.show()

    max_error = np.max(np.abs(y_deriv2_custom.numpy() - y_deriv2_scipy))
    print(f"Maximum absolute error (2nd derivative): {max_error}")

if __name__ == "__main__":
    print("Testing B-spline derivatives...")
    test_bspline_derivatives()
