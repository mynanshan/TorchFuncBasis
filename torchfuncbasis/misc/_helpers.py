'''
Computational helpers
'''


import torch
from torch import Tensor

def isequal_tensor(x: Tensor, y: Tensor) -> bool:
    return (
        x.shape == y.shape
        and bool(torch.all(torch.isclose(x, y)))
    )
