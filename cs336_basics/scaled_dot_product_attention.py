from math import sqrt
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor, device, dtype
from einops import reduce, einsum

from cs336_basics.softmax import softmax


class ScaledDotProductAttention(nn.Module):
    def __init__(self, device: device | None = None, dtype: dtype | None = None):
        super().__init__()

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k)
        masked_qk = qk
        if mask is not None:
            masked_qk[~mask] = float("-inf")
        sm = softmax(masked_qk, dim=-1)
        return einsum(sm, V, "... queries keys , ... values d_v ->  ... queries d_v")
