import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor, device, dtype


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device: device | None = None, dtype: dtype | None = None
    ):
        super().__init__()
        weights = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weights)
        self.weights = nn.Parameter(weights)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, "... d_model"]:
        return self.weights[token_ids]
