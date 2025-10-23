import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor, device, dtype

from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        d_ff: int,
        num_heads: int,
        rope_theta: float,
        rope_submodule: RotaryPositionalEmbedding | None = None,
        device: device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        if rope_submodule is None:
            head_dim = d_model // num_heads
            self.rope = RotaryPositionalEmbedding(rope_theta, head_dim, context_length, device=device, dtype=dtype)
        else:
            self.rope = rope_submodule

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        in_features = self.token_embeddings.forward(in_indices)

        x = in_features
        for layer in self.layers:
            x = layer(x)

        rms = self.ln_final.forward(x)

        return self.lm_head.forward(rms)
