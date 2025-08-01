import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor, device, dtype

from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: device | None = None, dtype: dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1x = self.w1.forward(x)
        silu_res = w1x * torch.sigmoid(w1x)
        w3x = self.w3.forward(x)
        w2in = silu_res * w3x
        return self.w2.forward(w2in)
