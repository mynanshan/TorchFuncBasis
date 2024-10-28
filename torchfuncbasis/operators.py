"""
Basis-related utilities.
"""

from typing import Sequence, Optional, Any
import torch
from torch import Tensor
from jaxtyping import Float
from .misc._defaults import DefaultDevice, DefaultTensorFloatType
from .basis._basis import Basis


def penalty_matrix(
    basis: Basis, pen_orders: int | Sequence[int] = 2,
    device: Optional[Any] = DefaultDevice,
    dtype: Optional[Any] = DefaultTensorFloatType,
) -> Float[Tensor, "n_basis n_basis"]:
    """Compute the penalty matrix of a basis."""

    if isinstance(pen_orders, int):
        pen_orders = (pen_orders, )
    pen_orders = tuple(pen_orders)
    nderivs = len(pen_orders)

    for d in pen_orders:
        if not isinstance(d, int) or d < 0:
            raise ValueError(
                "pen_orders must be a sequence of non-negative integers, "
                f"but got {pen_orders} instead"
            )

    n_basis = basis.n_basis
    penmat = torch.zeros(nderivs, n_basis, n_basis, dtype=dtype, device=device)

    for i, d in enumerate(pen_orders):
        penmat[i] = basis.gram_matrix(derivative=d, dtype=dtype, device=device)

    return penmat, pen_orders
