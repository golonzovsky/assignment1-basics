from math import sqrt
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor, device, dtype, mul
from einops import reduce, einsum, rearrange, repeat

from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_submodule: RotaryPositionalEmbedding,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope_submodule

        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, rope_submodule=rope_submodule)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, "... batch sequence_length d_model"],
    ) -> Float[Tensor, "... batch sequence_length d_model"]:
        rms1 = self.ln1.forward(in_features)

        # Build Int[Tensor, "... sequence_length"]
        *prefix, batch, seq_len, _ = in_features.shape
        token_positions = torch.arange(seq_len, device=in_features.device, dtype=torch.long)
        if len(prefix) > 0:
            token_positions = token_positions.expand(*prefix, seq_len)
        sa = in_features + self.attn.forward(in_features=rms1, token_positions=token_positions)

        rms2 = self.ln2.forward(sa)
        ff = sa + self.ffn.forward(x=rms2)

        return ff
