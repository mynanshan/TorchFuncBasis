import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from torchfuncbasis.basis._bspline import B_spline

# Test 1: Single curve
def test_single_curve():

    k = 3  # cubic spline
    nbreaks = 9

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

    # Compute spline using custom implementation
    y_custom = B_spline(x, c, t, k=k)
    print(f"y_custom shape: {y_custom.shape}")
    print(f"y_custom vals: {y_custom[-10:].tolist()}")

    # Compute spline using SciPy
    tck = (t.numpy(), c.numpy(), k)
    y_scipy = interpolate.splev(x.numpy(), tck)

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(x.numpy(), y_custom.numpy(), 'b-', label='Custom')
    plt.plot(x.numpy(), y_scipy, 'r--', label='SciPy')
    plt.legend()
    plt.title('Single Curve Comparison')
    plt.show()

    # Compute error
    max_error = np.max(np.abs(y_custom.numpy() - y_scipy))
    print(f"Maximum absolute error (single curve): {max_error}")

# Test 2: Batched curves
def test_batched_curves():
    k = 3  # cubic spline
    nbreaks = 9
    batch_size1 = 3
    batch_size2 = 2

    # Generate knots base
    t_base = torch.linspace(0, 1, nbreaks)
    # Pad the knots
    pad_left = torch.full((k,), t_base[0].item())
    pad_right = torch.full((k,), t_base[-1].item())
    t_base = torch.cat([pad_left, t_base, pad_right])

    # Expand to batch dimensions
    t = t_base.expand(batch_size1, batch_size2, -1)

    k = 3
    n_coef = nbreaks + k - 1
    c = torch.rand(batch_size1, batch_size2, n_coef)

    x = torch.linspace(0, 1, 100).expand(batch_size1, batch_size2, -1)

    # Compute spline using custom implementation
    print(f"x shape: {x.shape}")
    print(f"c shape: {c.shape}")
    print(f"t shape: {t.shape}")
    y_custom = B_spline(x, c, t, k=k)

    # Compute spline using SciPy for the first curve
    tck = (t[0, 0].numpy(), c[0, 0].numpy(), k)
    y_scipy = interpolate.splev(x[0, 0].numpy(), tck)

    # Plot comparison for the first curve
    plt.figure(figsize=(10, 5))
    plt.plot(x[0, 0].numpy(), y_custom[0, 0].numpy(), 'b-', label='Custom')
    plt.plot(x[0, 0].numpy(), y_scipy, 'r--', label='SciPy')
    plt.legend()
    plt.title('First Curve from Batch Comparison')
    plt.show()

    # Compute error for the first curve
    max_error = np.max(np.abs(y_custom[0, 0].numpy() - y_scipy))
    print(f"Maximum absolute error (first curve from batch): {max_error}")

    # Verify batch shape
    print(f"Output shape: {y_custom.shape}")

if __name__ == "__main__":
    # print("Testing single curve...")
    # test_single_curve()

    print("\nTesting batched curves...")
    test_batched_curves()
