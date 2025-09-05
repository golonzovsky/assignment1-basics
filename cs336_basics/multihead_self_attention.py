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

        self.initParam("wq")
        self.initParam("wk")
        self.initParam("wv")
        self.initParam("wo")

        self.attn = ScaledDotProductAttention()

    def initParam(self, name: str):
        w = torch.empty(self.d_model, self.d_model, device=self.device, dtype=self.dtype)
        torch.nn.init.trunc_normal_(w)
        # nn.init.xavier_uniform_(w)
        self.register_parameter(name, nn.Parameter(w))

    def forward(
        self,
        in_features: Float[Tensor, " ... seq d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... seq d_out"]:
        # Project first
        q_flat = einsum(in_features, self.wq, "... seq d_in, d_q d_in -> ... seq d_q")
        k_flat = einsum(in_features, self.wk, "... seq d_in, d_k d_in -> ... seq d_k")
        v_flat = einsum(in_features, self.wv, "... seq d_in, d_v d_in -> ... seq d_v")

        # Split into heads using rearrange
        q = rearrange(q_flat, "... seq (n_heads d_head) -> ... n_heads seq d_head", n_heads=self.num_heads)
        k = rearrange(k_flat, "... seq (n_heads d_head) -> ... n_heads seq d_head", n_heads=self.num_heads)
        v = rearrange(v_flat, "... seq (n_heads d_head) -> ... n_heads seq d_head", n_heads=self.num_heads)

        if self.rope is not None and token_positions is not None:
            print(f"!!!!!!!!!!!!{q.shape=} {in_features.shape=} {token_positions.shape=}")
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        # Apply attention
        mask = None
        if self.masked:
            seq_len = in_features.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()

        attention_output = self.attn(q, k, v, mask)
        multihead = rearrange(attention_output, "... n_heads seq d_head -> ... seq (n_heads d_head)")

        return einsum(multihead, self.wo, "... seq d_in, d_out d_in -> ... seq d_out")
