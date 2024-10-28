from typing import Optional, Any
import torch
from torch import Tensor
from jaxtyping import Float
from torchfuncbasis.misc._typing import TensorLike
from torchfuncbasis.misc._validate import validate_domain_range
from torchfuncbasis.misc._typing import DomainRange1D, DomainRangeLike
from torchfuncbasis.misc._typing import PointsND
from torchfuncbasis.misc._typing import BasisMat
from torchfuncbasis.misc._defaults import DefaultDevice, DefaultTensorFloatType
from torchfuncbasis.misc._helpers import isequal_tensor
from ._basis import Basis
from ._bspline import B_basis, B_basis_deriv



class BSplineBasis(Basis):
    r"""BSpline basis.

    BSpline basis elements are defined recursively as:

    .. math::
        B_{i, 1}(x) = 1 \quad \text{if } t_i \leq x < t_{i+1},
        \quad 0 \text{ otherwise}

    .. math::
        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
        + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    Where k indicates the order of the spline.

    Parameters:
        domain_range: A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis: Number of functions in the basis.
        order: Order of the splines. One greather than their degree.
        knots: List of knots of the spline functions.
    """

    def __init__(
        self, *,
        n_basis: Optional[int] = 3,
        domain_range: Optional[DomainRangeLike] = None,
        order: int = 4,
        knots: Optional[TensorLike] = None,
        dtype: Optional[torch.dtype] = DefaultTensorFloatType,
        device: Optional[torch.device] = DefaultDevice,
    ) -> None:
        """Bspline basis constructor."""

        # format the domain_range
        domain_range = validate_domain_range(domain_range)[0]

        # validate order
        if (not isinstance(order, int)) or order < 1:
            raise ValueError(
                "order has to be a positive integer, "
                f"but got order={order} instead"
            )

        # validate knots and n_basis
        if knots is not None:
            knots, _ = torch.as_tensor(knots).reshape(-1).sort()

        if n_basis is None:
            if knots is None:
                raise ValueError(
                    "Must provide either a list of knots or the"
                    "number of basis.",
                )
            else:
                # n_basis default to n_knots + order - 2
                n_basis = knots.numel() + order - 2
        else:
            if n_basis < order:
                raise ValueError(
                    f"The number of basis ({n_basis}) should not be smaller "
                    f"than the order of the bspline ({order}).",
                )
            if knots is None:
                # knots detault to an evenlly spaced grid in the domain
                knots = torch.linspace(
                    domain_range[0], domain_range[1],
                    n_basis - order + 2
                )
            else:
                # compatibility with domain_range and n_basis
                if not isequal_tensor(domain_range[0], knots[0]) or \
                    not isequal_tensor(domain_range[1], knots[-1]):
                    raise ValueError(
                        "The ends of the knots must be the same "
                        "as the domain_range.",
                    )
                if n_basis != order + knots.numel() - 2:
                    raise ValueError(
                        f"The number of basis ({n_basis}) has to "
                        f"equal the order ({order}) plus the "
                        f"number of knots ({knots.numel()}) minus 2.",
                    )

        super().__init__(n_basis=n_basis, domain_range=domain_range,
                         dtype=dtype, device=device)

        if self.dim_domain != 1:
            raise ValueError(
                "BSplineBasis is defined on an 1D domain, but "
                "domain_range does not match the dimension."
            )

        self._order = order
        self._knots = knots.to(dtype=dtype, device=device)


    @property
    def domain_range(self) -> DomainRange1D:
        """Range of basis function's domain"""
        return self._domain_range[0]

    @property
    def knots(self) -> Float[Tensor, "n_knots"]:
        """Returns the knot points of the B-spline basis."""
        return self._knots

    @property
    def n_knots(self) -> int:
        """Returns the number of knot points of the B-spline basis."""
        return self.knots.numel()

    @property
    def order(self) -> int:
        """Returns the order of the B-spline basis."""
        return self._order

    def _pad_knots(self) -> Float[Tensor, "{self.n_knots}+{self.order}-2"]:
        """
        Get the knots adding m knots to the boundary.

        This needs to be done in order to allow a discontinuous behaviour
        at the boundaries of the domain (see references).

        NOTE:
        ------
            We add small tranlation to boundary knots to make sure that knots do
            not overlap. This is not ideal. May changed to a better algorithm.

        """
        t = self.knots
        k = self.order - 1
        return torch.cat([
                torch.full((k,), t[0].item()),
                t,
                torch.full((k,), t[-1].item())
            ])

    def _evaluate(
        self,
        eval_points: PointsND,
        derivative: int = 0
    ) -> BasisMat:
        """Evaluate the B-spline basis functions at given points."""
        dtype = eval_points.dtype
        device = eval_points.device

        eval_points = eval_points.squeeze(-1) # drop the domain dimension

        # Extend knots with repeats at the ends
        extended_knots = self._pad_knots().to(dtype=dtype, device=device)

        if derivative > 0:
            return B_basis_deriv(
                eval_points, extended_knots, self.order - 1, derivative)

        return B_basis(eval_points, extended_knots, self.order - 1)

    def _gram_matrix(
        self, derivative: int,
        dtype: Optional[torch.dtype] = DefaultTensorFloatType,
        device: Optional[torch.device] = DefaultDevice,
    ) -> Float[Tensor, "n_basis n_basis"]:
        """Compute the Gram matrix of the B-spline basis."""
        nt = 1001
        eval_points = torch.linspace(
            self.domain_range[0], self.domain_range[1], nt,
            dtype=dtype, device=device
        )

        basis_matrices = self(eval_points, derivative=derivative)  # (nt, n_basis)
        dt = (
            (eval_points[-1] - eval_points[0]) / (nt - 1)
        ).item()  # dt for trapezoid rule

        gram = torch.trapezoid(
            torch.matmul(basis_matrices.unsqueeze(-1), basis_matrices.unsqueeze(-2)),
            dx=dt, dim=0
        ) # (n_basis, n_basis)
        return gram

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "Basis":
        """Change the type and/or device of a Basis."""
        dtype = dtype if dtype is not None else self.dtype
        device = device if device is not None else self.device
        self.dtype = dtype
        self.device = device

        self._knots = self._knots.to(dtype=dtype, device=device)

        if self._gram_matrix_cached is not None:
            self._gram_matrix_cached = self._gram_matrix_cached.to(
                dtype=dtype, device=device)

        return self

    def __repr__(self) -> str:
        """Representation of a BSpline basis."""
        return (
            f"{self.__class__.__name__}(domain_range={self.domain_range}, "
            f"n_basis={self.n_basis}, order={self.order}, "
            f"knots={self.knots})"
        )

    def __str__(self) -> str:
        """Print output of a BSpline basis."""
        return (
            f"{self.__class__.__name__}(domain_range={self.domain_range}, "
            f"n_basis={self.n_basis}, order={self.order}"
        )

    def __eq__(self, other: Any) -> bool:
        """Check if two BSplineBasis objects are equal."""
        return (
            super().__eq__(other)
            and self.order == other.order
            and isequal_tensor(self.knots, other.knots)
        )

    def __hash__(self) -> int:
        """Hash the BSplineBasis object."""
        return hash((
            super().__hash__(), self.order,
            tuple(self.knots.tolist())
        ))
