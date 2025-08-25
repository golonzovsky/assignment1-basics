import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor, device, dtype
from einops import reduce


def softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_vals = in_features.max(dim=dim, keepdim=True).values
    normalized = in_features - max_vals
    exp = torch.exp(normalized)
    exp_sum = exp.sum(dim=dim, keepdim=True)
    return exp / exp_sum
