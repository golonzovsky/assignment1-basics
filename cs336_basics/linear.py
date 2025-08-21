import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor, device, dtype
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: device | None = None, dtype: dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w)
        self.register_parameter("weights", nn.Parameter(w))


    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
