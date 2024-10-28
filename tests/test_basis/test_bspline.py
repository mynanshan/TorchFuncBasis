import pytest
import torch
from torchfuncbasis.basis import BSplineBasis


def test_bspline_creation():
    basis = BSplineBasis(n_basis=5, domain_range=(0, 1))
    assert basis.n_basis == 5
    assert torch.allclose(basis.domain_range, torch.tensor([0., 1.]))


def test_bspline_evaluation():
    basis = BSplineBasis(n_basis=5, domain_range=(0, 1))
    x = torch.linspace(0, 1, 100).unsqueeze(-1)
    values = basis(x)
    assert values.shape == (100, 5)


def test_bspline_derivatives():
    basis = BSplineBasis(n_basis=5, domain_range=(0, 1))
    x = torch.linspace(0, 1, 100).unsqueeze(-1)
    derivs = basis(x, derivative=1)
    assert derivs.shape == (100, 5)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_different_orders(order):
    basis = BSplineBasis(n_basis=5, domain_range=(0, 1), order=order)
    assert basis.order == order
