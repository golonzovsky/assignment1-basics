import torch.nn as nn
import torch
from torch import Tensor, device, dtype


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: device | None = None, dtype: dtype | None = None):
        super().__init__()
        # torch.nn.init.trunc_normal_
        w = torch.empty(out_features, in_features)
        torch.nn.init.trunc_normal_(w)
        self.w = nn.Parameter(w)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w.T
