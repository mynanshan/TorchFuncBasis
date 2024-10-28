# TorchFuncBasis

A PyTorch-based library for functional basis representations and smooth function approximation.

## Features

- B-spline basis functions with arbitrary order
- Fourier basis functions
- Smooth function approximation with penalized regression
- Batch processing support
- GPU acceleration through PyTorch

## Installation

```bash
pip install torchfuncbasis
```

## Quick Start

```python
import torch
from torchfuncbasis.basis import BSplineBasis
from torchfuncbasis.smoother import points2basiscoefs
```

### Create a B-spline basis
```python
basis = BSplineBasis(n_basis=11, domain_range=(0, 1), order=4)
print(basis)
```

### Evaluate Basis matrix or Gram matrix
A `Basis` object can be called to evaluate the basis matrix or its derivatives. The input is allowed to be a batch of points with shape `(*batch, n_points, *dim_domain)`:
* `batch` is arbitrary leading dimensions, can be empty
* `n_points` is the number of points in the domain
* `dim_domain` is the dimension of the domain, can be dropped if the domain is one-dimensional

```python
x = torch.linspace(0, 1, 101).unsqueeze(0).repeat(5, 7, 1)
print(f"x's shape: {x.shape}")
basis_matrix = basis(x)
print(f"basis_matrix.shape: {basis_matrix.shape}")
basis_deriv_matrix = basis(x, derivative=1)
gram_matrix = basis.gram_matrix()
print(f"gram_matrix.shape: {gram_matrix.shape}")
gram_matrix_deriv = basis.gram_matrix(derivative=1)
```

### Fit smooth function
Suppose we have a set of points `(x, y)` and a basis object `basis`. The `y` is expected to have shape `(*batch, n_points, *dim_response)`, where
* `*batch` and `n_points` are the same as those in `x`
* `dim_response` is the dimension of the response variable, can be dropped if the response is one-dimensional

We can fit a smooth function by solving a penalized least-squares problem with `points2basiscoefs`. The returned value is the basis coefficients of shape `(*batch, dim_response, n_basis)`.

```python
x = torch.linspace(0, 1, 101).unsqueeze(0).repeat(5, 7, 1)
print(f"x's shape: {x.shape}")
y = torch.cat([
    (torch.sin(2 * torch.pi * x) + torch.randn_like(x) * 0.05).unsqueeze(-1),
    (torch.cos(2 * torch.pi * x) + torch.randn_like(x) * 0.05).unsqueeze(-1)
], dim=-1)
print(f"y's shape: {y.shape}")
coefs = points2basiscoefs(x, y, basis, smoothing_param=1e-4)
print(f"coefs.shape: {coefs.shape}")
```

## References

- [scikit-fda](https://github.com/GAA-UAM/scikit-fda): a Numpy-based library for comprehensive functional data analysis, including basis functions and smoothing methods. `scikit-fda` is our main reference for code architecture.
- [pykan/spline.py](https://github.com/KindXiaoming/pykan/blob/master/kan/spline.py) for PyTorch-based B-spline implementation.


## To Be Added...
- Other univariate basis functions
- Tensor-product basis functions
- A wrapper for smoothing:w
- 