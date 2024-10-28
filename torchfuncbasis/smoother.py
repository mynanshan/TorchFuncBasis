"""
Smoothing discrete data with basis functions.
"""

from typing import Sequence
import torch
from torch import Tensor
from jaxtyping import Float
from torchfuncbasis.basis._basis import Basis
from torchfuncbasis.operators import penalty_matrix
from torchfuncbasis.misc._typing import TensorLike, PointsLike, ValuesLike
from torchfuncbasis.misc._validate import validate_evaluation_points, validate_response_values






def points2basiscoefs(
    x: PointsLike,
    y: ValuesLike,
    basis: Basis,
    penalty_orders: int | Sequence[int] = 2,
    smoothing_param: float | TensorLike = 1.0,
) -> Float[Tensor, "*batch n_channels n_basis"]:
    """
    Smooth points to a function in a basis representation.
    Inputs:
        x : PointsLike,
            validated shape (*batch, n_points, dim_domain)
        y : ValuesLike,
            validated shape (*batch, n_points, n_channels)
        basis: Basis
        penalty_orders: int | Sequence[int], validated shape (nderivs,)
        smoothing_param: float | TensorLike, validated shape (nderivs,)
    """

    x = validate_evaluation_points(x, dim_domain=basis.dim_domain)
    dtype = x.dtype
    device = x.device

    y = validate_response_values(y, points_dims=x.shape[:-1])
    y = y.to(dtype=dtype, device=device)

    basismat = basis(x)  # (*batch, n_points, n_basis)
    penmat, penalty_orders = penalty_matrix(
        basis, penalty_orders, dtype=dtype, device=device)  # (nderiv, n_basis, n_basis)

    nderivs = len(penalty_orders)
    smoothing_param = torch.as_tensor(
        smoothing_param, dtype=dtype, device=device).reshape(-1)
    if nderivs > 1 and smoothing_param.numel() == 1:
        smoothing_param = smoothing_param.repeat(nderivs)
    if smoothing_param.numel() != nderivs:
        raise ValueError(
            "smoothing_param must have the same length as penalty_orders."
        )

    penmat = (smoothing_param.reshape((-1, 1, 1)) * penmat).sum(dim=0)  # (n_basis, n_basis)

    # Compute Φ^T Φ + Ω_λ and its inverse
    gram_matrix = torch.einsum(
        '...np,...nq->...pq', basismat, basismat)  # (*batch, n_basis, n_basis)
    system_matrix = gram_matrix + penmat  # (*batch, n_basis, n_basis)

    # Solve the system: C = Y^T Φ (Φ^T Φ + Ω_λ)^{-1}
    rhs = torch.einsum(
        '...nc,...np->...cp', y, basismat)  # (*batch, n_channels, n_basis)
    coefs = torch.linalg.solve(
        system_matrix, rhs.transpose(-1, -2))  # (*batch, n_basis, n_channels)
    coefs = coefs.transpose(-1, -2)  # (*batch, n_channels, n_basis)

    return coefs
