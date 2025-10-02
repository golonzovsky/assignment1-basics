import numpy as np
import numpy.typing as npt
import torch
import os
from typing import IO, BinaryIO


def data_loader(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.choice(max_start_idx, size=batch_size, replace=False)

    sequences = np.array([dataset[i : i + context_length + 1] for i in start_indices])

    x = sequences[:, :-1]
    y = sequences[:, 1:]

    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)

    # print(f"{batch_size=} {context_length=} {sequences.shape=} {x.shape=} {y.shape=}")
    return (x_tensor, y_tensor)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state = {"model_weights": model.state_dict(), "optimizer_state": optimizer.state_dict(), "iteration": iteration}
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model_weights"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["iteration"]
