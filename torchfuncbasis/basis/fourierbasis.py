from typing import Optional, Any
import torch
from torch import Tensor
from jaxtyping import Float
from torchfuncbasis.misc._typing import DomainRange1D, DomainRangeLike
from torchfuncbasis.misc._typing import PointsND
from torchfuncbasis.misc._typing import BasisMat
from torchfuncbasis.misc._defaults import DefaultDevice, DefaultTensorFloatType
from torchfuncbasis.misc._helpers import isequal_tensor

from ._basis import Basis


class FourierBasis(Basis):
    r"""Fourier basis.

    Defines a functional basis for representing functions on a fourier
    series expansion of period :math:`T`. The number of basis is always odd.
    If instantiated with an even number of basis, they will be incremented
    automatically by one.

    .. math::
        \phi_0(t) = \frac{1}{\sqrt{2}}

    .. math::
        \phi_{2n -1}(t) = \frac{sin\left(\frac{2 \ pi n}{T} t\right)}
                                                    {\sqrt{\frac{T}{2}}}

    .. math::
        \phi_{2n}(t) = \frac{cos\left(\frac{2 \ pi n}{T} t\right)}
                                                    {\sqrt{\frac{T}{2}}}


    This basis will be orthonormal if the period coincides with the length
    of the interval in which it is defined.

    Parameters:
        domain_range: A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis: Number of functions in the basis.
        period: Period (:math:`T`).

    """

    def __init__(
        self,
        domain_range: Optional[DomainRangeLike] = None,
        n_basis: int = 3,
        period: Optional[float] = None,
        dtype: Optional[torch.dtype] = DefaultTensorFloatType,
        device: Optional[torch.device] = DefaultDevice,
    ) -> None:
        """
        Construct a FourierBasis object.

        It forces the object to have an odd number of basis. If n_basis is
        even, it is incremented by one.

        Args:
            domain_range: Tuple defining the domain over which the
                function is defined.
            n_basis: Number of basis functions.
            period: Period of the trigonometric functions that
                define the basis.

        """

        self._period = torch.as_tensor(period).squeeze().to(
            dtype=dtype, device=device)
        if self._period.numel() > 1:
            raise ValueError(
                "period should be a scalar. Got mutiple values instead.")
        # If number of basis is even, add 1
        n_basis += 1 - n_basis % 2
        super().__init__(n_basis=n_basis, domain_range=domain_range,
                         dtype=dtype, device=device)

    @property
    def domain_range(self) -> DomainRange1D:
        """Range of basis function's domain"""
        return self._domain_range[0]

    @property
    def period(self) -> Float[Tensor, ""]:
        """Period of the Fourier basis functions."""
        if self._period is None:
            return self.domain_range[1] - self.domain_range[0]

        return self._period

    def _evaluate(
        self,
        eval_points: PointsND,
        derivative: int = 0
    ) -> BasisMat:
        """Evaluate the Fourier basis functions at given points."""
        dtype = eval_points.dtype
        device = eval_points.device

        # Ensure eval_points is a tensor with shape (*batch, n)
        eval_points = eval_points.squeeze(-1)  # drop the domain dimension
        batch_shape = eval_points.shape[:-1]
        n = eval_points.shape[-1]

        # Compute omega and omegax
        period = self.period.to(dtype=dtype, device=device)
        omega = 2 * torch.pi / period
        omegax = omega * eval_points

        # Initialize the basis matrix
        basismat = torch.zeros(
            (*batch_shape, n, self.n_basis),
            dtype=dtype, device=device
        )

        sqrt_two_over_T = torch.sqrt(2.0 / period)

        if derivative == 0:
            # Compute the Fourier series itself
            basismat[..., 0] = 1.0 / torch.sqrt(period)
            if self.n_basis > 1:
                k = torch.arange(1, (self.n_basis - 1) // 2 +
                                 1, device=device, dtype=dtype)
                args = torch.einsum('...i,j->...ij', omegax, k)
                basismat[..., 1::2] = sqrt_two_over_T * torch.sin(args)
                basismat[..., 2::2] = sqrt_two_over_T * torch.cos(args)
        else:
            # Compute the derivative of order 'derivative'
            basismat[..., 0] = 0.0  # Derivative of constant term is zero
            if self.n_basis > 1:
                k = torch.arange(1, (self.n_basis - 1) // 2 +
                                 1, device=device, dtype=dtype)
                omega_k = k * omega
                coef = (omega_k ** derivative) * sqrt_two_over_T
                args = torch.einsum('...i,j->...ij', omegax, k)

                if derivative % 4 == 0:
                    basismat[..., 1::2] = coef * torch.sin(args)
                    basismat[..., 2::2] = coef * torch.cos(args)
                elif derivative % 4 == 1:
                    basismat[..., 1::2] = coef * omega_k * torch.cos(args)
                    basismat[..., 2::2] = -coef * omega_k * torch.sin(args)
                elif derivative % 4 == 2:
                    basismat[..., 1::2] = -coef * \
                        (omega_k ** 2) * torch.sin(args)
                    basismat[..., 2::2] = -coef * \
                        (omega_k ** 2) * torch.cos(args)
                elif derivative % 4 == 3:
                    basismat[..., 1::2] = -coef * \
                        (omega_k ** 3) * torch.cos(args)
                    basismat[..., 2::2] = coef * \
                        (omega_k ** 3) * torch.sin(args)

                # Adjust signs based on derivative order
                sign = (-1) ** ((derivative - 1) // 2)
                basismat[..., 1::2] *= sign
                basismat[..., 2::2] *= sign

        return basismat

    def _gram_matrix(
        self, derivative: int = 0,
        dtype: Optional[torch.dtype] = DefaultTensorFloatType,
        device: Optional[torch.device] = DefaultDevice,
    ) -> Float[Tensor, "n_basis n_basis"]:
        """
        Compute the Gram matrix for the Fourier basis.

        For derivatives, the Gram matrix elements are (k * omega)^{2 * derivative}
        where k is the basis index for sine and cosine terms.
        """
        n_basis = self.n_basis
        gram = torch.zeros(n_basis, n_basis, dtype=dtype, device=device)

        period = self.period.to(dtype=dtype, device=device)
        omega = 2 * torch.pi / period

        if derivative == 0:
            # For the base functions (no derivative), the Gram matrix is the identity matrix
            gram.fill_diagonal_(1.0)
        else:
            # Set the Gram matrix for derivatives
            # The derivative of the constant term is zero, so its norm is zero
            gram[0, 0] = 0.0

            if self.n_basis > 1:
                k = torch.arange(1, (self.n_basis - 1) // 2 +
                                 1, dtype=dtype, device=device)

                # Compute scaling factors
                scaling_factors = (omega * k) ** (2 * derivative)

                # Assign scaling factors to diagonal elements for sine and cosine terms
                diag_indices = torch.arange(
                    1, n_basis, dtype=torch.long, device=device)

                # Repeat each scaling factor twice for sine and cosine pairs
                scaling_factors_repeated = scaling_factors.repeat_interleave(2)

                # Set diagonal elements
                gram[diag_indices, diag_indices] = scaling_factors_repeated

        return gram

    def __repr__(self) -> str:
        """Representation of a Fourier basis."""
        return (
            f"{self.__class__.__name__}("
            f"domain_range={self.domain_range}, "
            f"n_basis={self.n_basis}, "
            f"period={self.period})"
        )

    def __eq__(self, other: Any) -> bool:
        """Check if two FourierBasis objects are equal."""
        return (
            super().__eq__(other)
            and isequal_tensor(self.period, other.period)
        )

    def __hash__(self) -> int:
        """Hash the FourierBasis object."""
        return hash((super().__hash__(), self.period))
