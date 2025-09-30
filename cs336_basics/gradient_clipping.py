import torch
import math
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor, device, dtype
from einops import reduce
from collections.abc import Iterable, Iterator


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:
    l2_norm_squared = 0.0
    for p in parameters:
        if p.grad is not None:
            l2_norm_squared += p.grad.data.pow(2).sum().item()
    l2_norm = math.sqrt(l2_norm_squared)

    if l2_norm < max_l2_norm:
        return

    for p in parameters:
        if p.grad is not None:
            p.grad.data.mul_(max_l2_norm / (l2_norm + eps))
