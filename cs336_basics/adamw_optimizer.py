import torch
from collections.abc import Callable
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, betas=None):
        if betas is None:
            betas = (0.9, 0.999)
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            b1 = group["betas"][0]
            b2 = group["betas"][1]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)  # First moment
                    state["v"] = torch.zeros_like(p.data)  # Second moment

                grad = p.grad.data
                t = state["t"]
                m = state["m"]
                v = state["v"]

                t += 1
                m.mul_(b1).add_(grad, alpha=(1 - b1))  # update first moment estimate
                v.mul_(b2).add_(grad * grad, alpha=(1 - b2))  # update the second moment estimate
                at = lr * math.sqrt(1 - math.pow(b2, t)) / (1 - math.pow(b1, t))  # bias-corrected lr

                p.data -= at * m / v.sqrt().add_(eps)
                p.data -= lr * weight_decay * p.data  # weight decay

                state["t"] = t

        return loss
