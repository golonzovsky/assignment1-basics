import numpy as np
import numpy.typing as npt
import torch
import pathlib
import os
import time
import argparse
from typing import IO, BinaryIO
import wandb

from cs336_basics.tokenize_dataset import load_tokenized_dataset
from cs336_basics.transformer import Transformer
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_scheduler import get_lr_cosine_shedule
from cs336_basics.adamw_optimizer import AdamW


# python -m cs336_basics.training \
#     --vocab-size 10000 \
#     --batch-size 32 \
#     --max-iters 1000 \
#     --wandb-project "cs336-pool" \
#     --wandb-run-name "alex-tinystories-first-attempt"

def data_loader(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.choice(max_start_idx, size=batch_size, replace=False)

    sequences = np.array([dataset[i : i + context_length + 1] for i in start_indices])

    x = sequences[:, :-1]
    y = sequences[:, 1:]

    x_tensor = torch.from_numpy(x).long().to(device)
    y_tensor = torch.from_numpy(y).long().to(device)

    return x_tensor, y_tensor


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

def get_default_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def train(
    dataset_path: str,
    vocab_size: int,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 6,
    d_ff: int = 2048,
    num_heads: int = 8,
    rope_theta: float = 10000.0,
    batch_size: int = 32,
    max_learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    warmup_iters: int = 100,
    weight_decay: float = 0.1,
    max_iters: int = 10000,
    eval_interval: int = 100,
    save_interval: int = 1000,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    device: str | None = None,
    grad_clip: float = 1.0,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):

    # Setup
    if device is None:
        device = get_default_device()
    pathlib.Path(checkpoint_dir).mkdir(exist_ok=True)
    print(f"Training on device: {device}")

    # Initialize Weights & Biases
    if wandb_project:
        config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "num_heads": num_heads,
            "rope_theta": rope_theta,
            "batch_size": batch_size,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_iters": warmup_iters,
            "weight_decay": weight_decay,
            "max_iters": max_iters,
            "grad_clip": grad_clip,
            "device": device,
        }
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_tokenized_dataset(dataset_path)
    print(f"Dataset size: {len(dataset):,} tokens")

    # Initialize model
    print("Initializing model...")
    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        num_heads=num_heads,
        rope_theta=rope_theta,
        device=device,
    )

    # Explicitly move model to device (important for MPS)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    if wandb_project:
        wandb.config.update({"num_params": num_params})

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
    )

    # Resume from checkpoint if specified
    start_iter = 0
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        start_iter = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Training loop
    print(f"\nStarting training from iteration {start_iter}...")
    model.train()

    losses = []
    start_time = time.time()

    for iter_num in range(start_iter, max_iters):
        # Update learning rate based on schedule
        lr = get_lr_cosine_shedule(iter_num, max_learning_rate, min_learning_rate, warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Sample batch
        x, y = data_loader(dataset, batch_size, context_length, device)

        # Forward pass
        logits = model(x)  # [batch_size, seq_len, vocab_size]

        # Compute loss
        loss = cross_entropy(
            logits.reshape(-1, vocab_size),  # [batch_size * seq_len, vocab_size]
            y.reshape(-1),  # [batch_size * seq_len]
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            gradient_clipping(model.parameters(), grad_clip)

        optimizer.step()

        # Track loss
        losses.append(loss.item())

        # Log to wandb every step
        if wandb_project:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": lr,
                "train/iteration": iter_num,
            }, step=iter_num)

        # Logging
        if (iter_num + 1) % eval_interval == 0:
            avg_loss = sum(losses) / len(losses)
            elapsed = time.time() - start_time
            iter_per_sec = (iter_num + 1 - start_iter) / elapsed

            print(f"Iter {iter_num + 1}/{max_iters} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Speed: {iter_per_sec:.2f} iter/s")

            if wandb_project:
                wandb.log({
                    "train/avg_loss": avg_loss,
                    "train/iter_per_sec": iter_per_sec,
                }, step=iter_num)

            losses = []

        # Save checkpoint
        if (iter_num + 1) % save_interval == 0:
            checkpoint_path = pathlib.Path(checkpoint_dir) / f"checkpoint_{iter_num + 1}.pt"
            save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final checkpoint
    final_path = pathlib.Path(checkpoint_dir) / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, max_iters, final_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # Finish wandb
    if wandb_project:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer language model")

    # Data
    parser.add_argument("--dataset", type=str, default="tinystories_train_tokenized.bin",
                        help="Path to tokenized dataset")
    parser.add_argument("--vocab-size", type=int, required=True,
                        help="Vocabulary size")

    # Model architecture
    parser.add_argument("--context-length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--d-model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048,
                        help="Feed-forward dimension")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--rope-theta", type=float, default=10000.0,
                        help="RoPE base frequency")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--max-learning-rate", type=float, default=3e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min-learning-rate", type=float, default=3e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup-iters", type=int, default=100,
                        help="Number of warmup iterations")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--max-iters", type=int, default=10000,
                        help="Maximum training iterations")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping threshold")

    # Logging and checkpointing
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Log metrics every N iterations")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (cuda/mps/cpu, default: auto-detect)")

    # Weights & Biases
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Weights & Biases project name (if not set, wandb is disabled)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Weights & Biases run name")

    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=args.device,
        grad_clip=args.grad_clip,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
