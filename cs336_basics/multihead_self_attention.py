from math import sqrt
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor, device, dtype, mul
from einops import reduce, einsum, rearrange, repeat

from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.linear import Linear


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        masked: bool = True,
        rope_submodule: RotaryPositionalEmbedding | None = None,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.masked = masked
        self.rope = rope_submodule

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.attn = ScaledDotProductAttention()


    def forward(
        self,
        in_features: Float[Tensor, " ... seq d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... seq d_out"]:
        q_flat = self.q_proj(in_features)
        k_flat = self.k_proj(in_features)
        v_flat = self.v_proj(in_features)

        # Split into heads
        q = rearrange(q_flat, "... seq (n_heads d_head) -> ... n_heads seq d_head", n_heads=self.num_heads)
        k = rearrange(k_flat, "... seq (n_heads d_head) -> ... n_heads seq d_head", n_heads=self.num_heads)
        v = rearrange(v_flat, "... seq (n_heads d_head) -> ... n_heads seq d_head", n_heads=self.num_heads)

        if self.rope is not None and token_positions is not None:
            # print(f"!!!!!!!!!!!!{q.shape=} {in_features.shape=} {token_positions.shape=}")
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        mask = None
        if self.masked:
            seq_len = in_features.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()

        attention_output = self.attn(q, k, v, mask)
        multihead = rearrange(attention_output, "... n_heads seq d_head -> ... seq (n_heads d_head)")

        return self.output_proj(multihead)
