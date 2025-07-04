import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor, device, dtype
from einops import reduce


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: device | None = None, dtype: dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        w = torch.empty(d_model, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w)
        self.weights = nn.Parameter(w)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        result = x * self.weights / rms
        return result.to(in_dtype)
