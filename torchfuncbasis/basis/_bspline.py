'''
de, B. (1978). A practical guide to splines.
'''

import torch
from torch import Tensor
from jaxtyping import Float


def B_basis(
    x: Float[Tensor, "*batch n_points"],
    t: Float[Tensor, "*batch_t n_knots"],
    k: int = 3,
) -> Float[Tensor, "*batch n_points n_knots-{k}-1"]:
    '''
    evaludate x on B-spline bases

    Args:
    -----
        x : ND torch.tensor, evaluation points
            shape (*batch, n_points)
        t : sorted knots, probably including padded knots
            ND torch.tensor, shape (*batch_t, n_knots)
            batch_t must be broadcastable to batch
        k : int, degree of splines.

    Returns:
    --------
        basis values : ND torch.tensor
            shape (*batch, n_points, num basis).
            num basis = num knots - degree - 1
    '''

    # Reference:
    #     - https://github.com/KindXiaoming/pykan/blob/master/kan/spline.py

    assert x.ndim > 0, "x must have at least one dimension"
    assert t.numel() > 1, "knots must have more than one element"

    # insert one dim to x and t for broadcasted computation
    x = x.unsqueeze(-1) # (*batch_dims, n_points, 1)
    t = t.unsqueeze(-2) # (*batch_dims, 1, n_knots)

    if k == 0:
        # B_0, first order basis, right boundary included
        value = (
            (x >= t[..., :-1]) * (x < t[..., 1:]) +  # most terms
            (x == t[..., -1:]) * (t[..., 1:] == t[..., -1:])  # last basis on the right boundary
        )
    else:
        B_km1 = B_basis(x[..., 0], t=t[..., 0, :], k=k - 1)
        coefl = (x - t[..., :-(k + 1)]) / (t[..., k:-1] - t[..., :-(k + 1)])
        coefr = (t[..., k + 1:] - x) / (t[..., k + 1:] - t[..., 1:(-k)])
        coefl = torch.nan_to_num(coefl, 0.)
        coefr = torch.nan_to_num(coefr, 0.)
        value = coefl * B_km1[..., :-1] + coefr * B_km1[..., 1:]

    return value


def B_basis_deriv(
    x: Float[Tensor, "*batch n_points"],
    t: Float[Tensor, "*batch_t n_knots"],
    k: int = 3,
    deriv: int = 1
) -> Float[Tensor, "*batch n_basis"]:
    '''
    evaludate x on B-spline bases

    Args:
    -----
        x : ND torch.tensor, evaluation points
            shape (*batch, n_points)
        t : sorted knots, probably including padded knots
            ND torch.tensor, shape (*batch_t, n_knots)
            batch_t must be broadcastable to batch
        k : int, degree of splines.

    Details:
    --------
        A brutal solution:
        Use spline evaluation with identity coef matrix

    Returns:
    --------
        basis derivative values : ND torch.tensor
            shape (*batch, n_points, num basis).
            num basis = num knots - degree - 1
    '''
    n_knots = t.shape[-1]
    n_basis = n_knots - k - 1

    # basis dim as a batch dim
    x = x.unsqueeze(-2)
    t = t.unsqueeze(-2)

    deriv_coefs = _B_deriv_coefs(
        torch.eye(n_basis), t, k, deriv
    )
    basis_matrix = B_basis(x, t, k - deriv)

    # spline values in shape (*batch, n_basis, n_points)
    spline_values = torch.matmul(basis_matrix, deriv_coefs.unsqueeze(-1))[..., 0]

    # switch the basis dimension back to the last dim
    return torch.swapdims(spline_values, -1, -2)


def B_spline(
    x: Float[Tensor, "*batch n_points"],
    c: Float[Tensor, "*batch_c n_basis"],
    t: Float[Tensor, "*batch_t n_knots"],
    k: int = 3,
    deriv: int = 0
) -> Float[Tensor, "*batch n_points"]:
    '''
    evaludate x on B-spline bases

    Args:
    -----
        x : ND torch.tensor, evaluation points
            shape (*batch, n_points)
        c : ND torch.tensor, basis coefficients
            shape (*batch_c, n_basis)
            batch_c must be broadcastable to batch
        t : sorted knots, probably including padded knots
            ND torch.tensor, shape (*batch_t, n_knots)
        k : int, degree of splines.
        deriv : int, order of derivtive

    Returns:
    --------
        spline values : ND torch.tensor
            shape (*batch, n_points).
    '''

    if deriv == 0:
        basis_matrix = B_basis(x, t, k)  # (*batch, n_points, n_basis)
        spline_values = torch.matmul(basis_matrix, c.unsqueeze(-1))[..., 0]
    else:
        deriv_coefs = _B_deriv_coefs(c, t, k, deriv)
        basis_matrix = B_basis(x, t, k - deriv)
        spline_values = torch.matmul(basis_matrix, deriv_coefs.unsqueeze(-1))[..., 0]
    return spline_values


def _B_deriv_coefs(
    c: Float[Tensor, "*batch_c n_basis"],
    t: Float[Tensor, "*batch_t n_knots"],
    k: int = 3,
    m: int = 0
) -> Float[Tensor, "*batch n_knots-{k}-1+{m}"]:
    '''
    Compute coefficients of B-spline derivatives

    Args:
    -----
        c : ND torch.tensor, basis coefficients
            shape (*batch_c, n_basis)
            batch_c must be broadcastable to batch
        t : sorted knots, probably including padded knots
            ND torch.tensor, shape (*batch_t, n_knots)
        k : int, degree of splines.
        m : int, order of derivtive

    Returns:
    --------
        coefs : ND torch.tensor
            shape (*batch, n_basis_deriv).
            n_basis_deriv = n_knots - {k} - 1 + m
    '''

    coefs = c
    for p in range(1, m+1):
        kmp = k + 1 - p
        factor = (t[..., kmp:] - t[..., :-kmp]) / (k + 1 - p)
        coefs = torch.cat([
            torch.zeros_like(coefs[..., :1]),
            coefs,
            torch.zeros_like(coefs[..., -1:])
        ], dim=-1)
        coefs = (coefs[..., 1:] - coefs[..., :-1]) / factor
        coefs = torch.nan_to_num(coefs, 0.)

    return coefs
