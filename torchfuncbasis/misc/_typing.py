import torch
from torch import Tensor
from numpy import ndarray
from numpy.typing import ArrayLike
from typing import Union, Tuple, Sequence
from jaxtyping import Float

TensorLike = Union[Tensor, ArrayLike]


DomainRange1D = Float[Tensor, "2"]
DomainRangeND = Float[Tensor, "dim_domain 2"]
DomainRange = Union[DomainRange1D, DomainRangeND]
DomainRangeLike = Union[
    DomainRange,
    Tuple[float, float],
    Sequence[Tuple[float, float]],
    Sequence[Sequence[float]]
]

Points1D = Float[Tensor, "*batch n_points"]
PointsND = Float[Tensor, "*batch n_points dim_domain"]
Points = Union[Points1D, PointsND]
PointsLike = Union[Points, TensorLike]

Values1D = Float[Tensor, "*batch n_points"]
ValuesND = Float[Tensor, "*batch n_points n_channels"]
Values = Union[Values1D, ValuesND]
ValuesLike = Union[Values, TensorLike]

BasisMat = Float[Tensor, "*batch n_points n_basis"]