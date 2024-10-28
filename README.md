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
basis = BSplineBasis(n_basis=10, domain_range=(0, 1), order=4)
```

### Evaluate Basis matrix or Gram matrix
```python
basis_matrix = basis(x)
basis_deriv_matrix = basis(x, derivative=1)
gram_matrix = basis.gram_matrix(x)
gram_matrix_deriv = basis.gram_matrix(x, derivative=1)
```

### Fit smooth function
```python
x = torch.linspace(0, 1, 100).unsqueeze(-1)
y = torch.sin(2 torch.pi x) + torch.randn_like(x) 0.1
coefs = points2basiscoefs(x, y, basis, smoothing_param=1e-4)
```

