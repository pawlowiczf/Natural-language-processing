"""Shared training loop and evaluation for every architecture notebook.

    from train import fit, evaluate
    history = fit(model, steps=5000)

The model contract is the one from dataset_loading.py: model(x[B, T]) -> logits[B, T, vocab].
"""

import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from dataset_loading import EOT, get_batch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# mixed precision (float16) training on CUDA GPUs
USE_AMP = device.type == "cuda"


def make_param_groups(model: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    # no weight decay for biases and normalization parameters (1D tensors)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def lr_multiplier(
    step: int, total_steps: int, warmup_steps: int, schedule: str
) -> float:
    if step < warmup_steps:  # linear warmup 0 -> 1
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # cosine 1 -> 0.1


def autocast():
    return torch.autocast(device.type, dtype=torch.float16, enabled=USE_AMP)


@torch.no_grad()
def evaluate(
    model: nn.Module, split: str, block_size: int, batch_size: int, n_batches: int = 50
) -> float:
    """Mean next-token cross-entropy on `n_batches` random windows of `split` ("val" or "test")."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(split, batch_size, block_size, device)
        with autocast():
            logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1).float(), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def fit(
    model: nn.Module,
    steps: int,
    *,
    batch_size: int = 64,
    block_size: int = 256,
    lr: float = 6e-4,
    weight_decay: float = 0.1,
    schedule: str = "cosine",
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    eval_every: int = 500,
    eval_batches: int = 50,
    save_path: str | None = None,  # e.g. "data/gpt.pt"; saved once at the end; None = do not save
) -> dict[str, list[float]]:
    model = model.to(device)
    opt = torch.optim.AdamW(
        make_param_groups(model, lr, weight_decay), betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: lr_multiplier(step, steps, warmup_steps, schedule)
    )
    # scales the loss to avoid float16 underflow
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    history = {"step": [], "train_loss": [], "val_loss": [], "time": []}
    start, loss_sum, n = time.time(), 0.0, 0
    model.train()

    pbar = tqdm(range(1, steps + 1), desc="training")
    for step in pbar:
        x, y = get_batch("train", batch_size, block_size, device)
        with autocast():
            logits = model(x)
        # logits (B, T, V) and targets (B, T) -> one big list of next-token predictions
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # so that the clipping threshold is in real gradient units
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        loss_sum += loss.item()
        n += 1

        if step % eval_every == 0 or step == steps:
            val_loss = evaluate(model, "val", block_size, batch_size, eval_batches)
            history["step"].append(step)
            history["train_loss"].append(loss_sum / n)
            history["val_loss"].append(val_loss)
            history["time"].append(time.time() - start)
            tqdm.write(
                f"step {step:6d}/{steps} | train loss {loss_sum / n:.3f} | val loss {val_loss:.3f} "
                f"(ppl {math.exp(val_loss):.1f}) | {time.time() - start:.0f} s"
            )
            loss_sum, n = 0.0, 0
    if save_path:  # weights only, saved once training has finished (an interrupted run leaves the old file intact)
        torch.save(model.state_dict(), save_path)
    return history


@torch.no_grad()
def generate(
    model: nn.Module,
    tok,
    prompt: str,
    max_new_tokens: int = 100,
    *,
    block_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    stop_at_eot: bool = True,
) -> str:
    """Continue `prompt`: predict the next token, sample it, append it, repeat."""
    model.eval()
    ids = torch.tensor([tok.encode(prompt).ids], device=device)  # (1, T)
    assert ids.shape[1] > 0, "empty prompt"
    eot = tok.token_to_id(EOT)
    for _ in range(max_new_tokens):
        with autocast():
            logits = model(
                ids[:, -block_size:]
            )  # only the last block_size tokens fit in the model
        logits = (
            logits[:, -1, :].float() / temperature
        )  # scores for the NEXT token only: (1, V)
        if top_k:  # keep the k best candidates, forbid the rest
            kth_best = torch.topk(logits, top_k).values[:, -1:]
            logits[logits < kth_best] = float("-inf")
        next_id = torch.multinomial(
            logits.softmax(dim=-1), num_samples=1
        )  # sample, not argmax
        if stop_at_eot and next_id.item() == eot:
            break
        ids = torch.cat([ids, next_id], dim=1)
    model.train()
    return tok.decode(ids[0].tolist())
