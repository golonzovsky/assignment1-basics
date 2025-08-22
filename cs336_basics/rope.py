import torch
import math
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor, device, dtype, cos, sin, tensor
from einops import reduce, einsum


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: device | None = None, dtype: dtype | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        assert self.d_k % 2 == 0

        matrices = [self.rotation_matrix(i) for i in range(self.max_seq_len)]
        precomputed: Float[Tensor, "max_seq_len d_k d_k"] = torch.stack(matrices)
        self.register_buffer("precomputed", precomputed, persistent=False)

    def rotation_matrix(self, i: int) -> Float[Tensor, "d_k d_k"]:
        res = torch.zeros(self.d_k, self.d_k, device=self.device, dtype=self.dtype)
        for k in range(self.d_k // 2):
            idx = k * 2
            res[idx:idx + 2, idx:idx + 2] = self.rotation_block(i, k)
        # blocks.append(block) return torch.block_diag(*blocks)
        return res

    def rotation_block(self, i: int, k: int) -> Float[Tensor, "2 2"]:
        angle = i / (self.theta ** (2 * k / self.d_k))
        # Compute in log space for numerical stability
        # log_theta = math.log(self.theta)
        # angle = i * math.exp(-2 * k * log_theta / self.d_k)
        s = math.sin(angle) # consider using torch.sin vectorized
        c = math.cos(angle)
        return torch.tensor([[c, -s], [s, c]], device=self.device, dtype=self.dtype)

    def forward(
        self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        rotations = self.precomputed[token_positions]
        # print(f"{x.shape=}, {token_positions.shape=}, {self.precomputed.shape=}, {rotations.shape=}")
        return einsum(
            x,
            rotations,
            "... seq_len d_k, seq_len d_k2 d_k -> ... seq_len d_k2"
        )
