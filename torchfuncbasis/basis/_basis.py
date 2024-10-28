from abc import ABC, abstractmethod
from typing import Optional, Any
import torch
from torch import Tensor
from jaxtyping import Float
from torchfuncbasis.misc._validate import validate_domain_range, validate_evaluation_points
from torchfuncbasis.misc._typing import DomainRange, DomainRangeLike
from torchfuncbasis.misc._typing import PointsND, PointsLike
from torchfuncbasis.misc._typing import BasisMat
from torchfuncbasis.misc._defaults import DefaultDevice, DefaultTensorFloatType


# mimic skfda's Basis
class Basis(ABC):
    """
    Base class for basis functions.
    """

    def __init__(
        self, n_basis: Optional[int] = 3, *,
        domain_range: Optional[DomainRangeLike] = None,
        dtype: Optional[torch.dtype] = DefaultTensorFloatType,
        device: Optional[torch.device] = DefaultDevice,
    ):

        if n_basis is None:
            n_basis = 3
        if (not isinstance(n_basis, int)) or n_basis <= 0:
            raise ValueError(
                "n_basis has to be a positive integer, "
                f"but got n_basis={n_basis} instead"
            )

        self._n_basis = n_basis
        domain_range = validate_domain_range(domain_range)
        self._domain_range = domain_range.to(
            dtype=dtype, device=device
        )

        # cached values
        self._cached_objects = ['gram_matrix']
        self._gram_matrix_cached = None

        # factory args
        self.dtype = dtype
        self.device = device

    @property
    def n_basis(self) -> int:
        """Number of basis functions"""
        return self._n_basis

    @property
    def domain_range(self) -> DomainRange:
        """Range of basis function's domain"""
        if self.dim_domain == 1:
            return self._domain_range.squeeze()
        return self._domain_range

    @property
    def dim_domain(self) -> int:
        """Dimension of basis function's domain"""
        return self._domain_range.shape[0]

    def __call__(
        self,
        eval_points: PointsLike,
        *,
        derivative: int = 0
    ) -> BasisMat:
        """Evaluate Basis objects.
        Evaluates the basis functions at a tensor of given values.
        Args:
            eval_points: Tensor of points where the basis is evaluated.
        Returns:
            basis matrix: Tensor, shape (n_basis, n_points), where
                m = eval_points.numel
        """

        eval_points = validate_evaluation_points(
            eval_points,
            dim_domain=self.dim_domain
        )

        dtype = eval_points.dtype
        device = eval_points.device

        if derivative < 0 or (not isinstance(derivative, int)):
            raise ValueError("derivative only takes non-negative integers.")

        basismat = self._evaluate(eval_points, derivative=derivative)

        assert basismat.dtype == dtype
        assert basismat.device == device

        return basismat

    @abstractmethod
    def _evaluate(
        self,
        eval_points: PointsND,
        derivative: int = 0
    ) -> BasisMat:
        """
        Evaluate Basis object.
        Subclasses must override this to provide basis evaluation.
        TODO: careful of dtype and device
        """

    def gram_matrix(
        self, derivative: int = 0, cache: bool = False,
        dtype: Optional[torch.dtype] = DefaultTensorFloatType,
        device: Optional[torch.device] = DefaultDevice,
    ) -> Float[Tensor, "n_basis n_basis"]:
        r"""
        Return the Gram Matrix of a basis.

        The Gram Matrix is defined as

        .. math::
            G_{ij} = \langle\phi_i, \phi_j\rangle

        where :math:`\phi_i` is the ith element of the basis. This is a
        symmetric matrix and positive-semidefinite.

        Returns:
            Gram Matrix of the basis.
        """

        dtype = dtype if dtype is not None else self.dtype
        device = device if device is not None else self.device

        if derivative == 0:

            gram = self._gram_matrix(0, dtype, device)

            if cache and self._gram_matrix_cached is None:
                self._gram_matrix_cached = gram.to(
                    dtype=self.dtype, device=self.device)

            return gram

        else:

            if derivative < 0 or (not isinstance(derivative, int)):
                raise ValueError(
                    "derivative only takes non-negative integers.")

            deriv_gram = self._gram_matrix(derivative, dtype, device)

            # if cache: TODO

            return deriv_gram

    @abstractmethod
    def _gram_matrix(
        self, derivative: int,
        dtype: Optional[torch.dtype] = DefaultTensorFloatType,
        device: Optional[torch.device] = DefaultDevice,
    ) -> Float[Tensor, "n_basis n_basis"]:
        """
        Compute the Gram matrix.

        Subclasses may override this method for improving computation
        of the Gram matrix.

        """

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "Basis":
        """Change the type and/or device of a Basis."""
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        if self._gram_matrix_cached is not None:
            self._gram_matrix_cached = self._gram_matrix_cached.to(
                dtype=dtype, device=device)

        return self

    def clear_cache(self) -> None:
        """Clear cached values."""
        for name in self._cached_objects:
            cached_attr = f"_{name}_cached"
            if hasattr(self, cached_attr):
                setattr(self, cached_attr, None)

    def __repr__(self) -> str:
        """Representation of a Basis object."""
        return (
            f"{self.__class__.__name__}("
            f"domain_range={self.domain_range}, "
            f"n_basis={self.n_basis})"
        )

    def __eq__(self, other: Any) -> bool:
        """Test equality of Basis."""
        return (
            isinstance(other, type(self))
            and torch.equal(self._domain_range, other._domain_range)
            and self.n_basis == other.n_basis
        )

    def __hash__(self) -> int:
        """Hash a Basis."""
        return hash((self._domain_range, self.n_basis))
