""" Validate the inputs of the functions in torchfuncbasis. """


from typing import Optional, Sequence

import torch
from ._typing import DomainRangeND, DomainRangeLike
from ._typing import PointsND, PointsLike
from ._typing import ValuesND, ValuesLike

# rewritten basded on skfda
def validate_evaluation_points(
    eval_points: PointsLike,
    *,
    dim_domain: int = 1,
) -> PointsND:
    """Convert and reshape the eval_points.
    Args:
        eval_points: Evaluation points to be reshaped.
        dim_domain: Dimension of the domain.
    """
    eval_points = torch.as_tensor(eval_points)

    shape_check = (
        (eval_points.shape[-1] == dim_domain)
        or (dim_domain == 1)
    )

    if shape_check:
        if dim_domain == 1 and eval_points.shape[-1] != 1:
            eval_points = eval_points.unsqueeze(-1)
        batch_dims = eval_points.shape[:-1]
        shape = (*batch_dims, dim_domain)
    else:
        raise ValueError(
            "Invalid shape for evaluation points. "
            f"An array with size (*batch, dim_domain (={dim_domain})) "
            "was expected. "
            "Instead, the received evaluation points have shape "
            f"{eval_points.shape}.",
        )

    return eval_points.reshape(shape)


def validate_response_values(
    responses: ValuesLike,
    *,
    points_dims: Sequence[int] | torch.Size
) -> ValuesND:
    """Convert and reshape the response values.
    Args:
        responses: Response values to be reshaped.
        batch_dims: Batch dimensions.
    """
    responses = torch.as_tensor(responses)
    points_dims = torch.Size(points_dims)
    ndim_points = len(points_dims)

    if (
        responses.ndim < ndim_points
        or responses.shape[:ndim_points] != points_dims
    ):
        raise ValueError(
            "Invalid shape for response values. "
            f"An array with size {points_dims} was expected. "
            f"Instead, the received response values have shape {responses.shape}."
        )

    return responses.reshape(points_dims + (-1,))


def validate_domain_range(
    domain_range: Optional[DomainRangeLike] = None
) -> DomainRangeND:
    """Validate the domain range."""
    if domain_range is None:
        domain_range = torch.tensor((0., 1.))
    else:
        domain_range = torch.as_tensor(domain_range)
    domain_range = domain_range.squeeze()
    if (not domain_range.ndim in (1, 2)) or domain_range.shape[-1] != 2:
        raise ValueError(
            "Invalid shape of dim_domain. Must be a Tensor of shape "
            "(2,) or (dim_domain, 2), or can be converted to such Tensor."
        )
    if domain_range.ndim == 1:
        domain_range = domain_range.unsqueeze(0)
    return domain_range
