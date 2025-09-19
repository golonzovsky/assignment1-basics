import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # print(f"{inputs=} {targets=}")
    # reimplement softmax to cancel out nominator log(exp(...))
    max_vals = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_vals
    # print(f"{max_vals=} {shifted=}")

    exp_sum = torch.exp(shifted).sum(dim=-1, keepdim=True)
    log_sum_exp = exp_sum.log()

    batch_idx = torch.arange(inputs.size(0), device=inputs.device)
    p_true = shifted[batch_idx, targets.long()]
    # print(f"{batch_idx=} {p_true=}")
    return (log_sum_exp - p_true).mean()
