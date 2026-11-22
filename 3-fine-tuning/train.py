import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from tqdm.auto import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Mixed precision (float16) training on CUDA GPUs.
USE_AMP = device.type == "cuda"


def make_param_groups(model: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    # No weight decay for biases and normalization parameters (1D tensors).
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


def to_device(batch) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward(model: nn.Module, batch: dict) -> torch.Tensor:
    """Logits (B,) as float32, whatever the autocast precision was."""
    with autocast():
        return model(batch["input_ids"], batch["attention_mask"]).float()


@torch.no_grad()
def predict(
    model: nn.Module, loader, max_batches: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Logits and labels for a loader (the first `max_batches` batches, or all of them if None),
    plus the wall-clock time in seconds.
    The time includes tokenization in the collate_fn: it is the real cost of classifying raw text.
    """
    model.eval()
    logits, labels = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch)
        logits.append(forward(model, batch).cpu())
        labels.append(batch["labels"].cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(logits), torch.cat(labels), time.time() - start


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict:
    probs = logits.sigmoid().numpy()
    y = labels.numpy().astype(int)
    pred = (probs >= threshold).astype(int)
    return {
        "loss": F.binary_cross_entropy_with_logits(logits, labels).item(),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),  # F1 of the positive (toxic) class
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "auroc": roc_auc_score(y, probs),  # threshold-free
        "auprc": average_precision_score(y, probs),  # threshold-free
        "threshold": threshold,
    }


def best_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Decision threshold maximizing F1 of the toxic class. Tune it on validation, never on test."""
    probs = logits.sigmoid().numpy()
    y = labels.numpy().astype(int)
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, probs >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(
    model: nn.Module, loader, threshold: float = 0.5, max_batches: int | None = None
) -> dict:
    """
    Loss, accuracy, F1, macro-F1, AUROC, AUPRC and inference time on a loader (e.g. the test set).
    Only the first `max_batches` batches are used, or all of them if None.
    """
    logits, labels, elapsed = predict(model, loader, max_batches)
    metrics = compute_metrics(logits, labels, threshold)
    metrics["inference_time"] = elapsed
    metrics["n_samples"] = len(labels)
    return metrics


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    schedule: str = "cosine",
    grad_clip: float = 1.0,
    # Metric used to pick the best epoch: "auprc", "auroc", "f1_tuned" or "loss" (lower is better).
    monitor: str = "auprc",
    # Minimum improvement of the monitored metric that resets the patience counter.
    min_delta: float = 0.001,
    # Early stopping: epochs without improvement of the monitored metric.
    patience: int = 3,
    pos_weight: float
    | None = None,  # e.g. n_negative / n_positive to counter class imbalance
    save_path: str
    | None = None,  # best weights saved once at the end; None = do not save
) -> dict:
    if monitor not in ("auprc", "auroc", "f1_tuned", "loss"):
        raise ValueError(f"Unknown monitor: {monitor}")

    model = model.to(device)
    opt = torch.optim.AdamW(
        make_param_groups(model, lr, weight_decay), betas=(0.9, 0.95)
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: lr_multiplier(step, total_steps, warmup_steps, schedule)
    )
    scaler = torch.amp.GradScaler(
        enabled=USE_AMP
    )  # scales the loss to avoid float16 underflow
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=device)

    history = {
        k: []
        for k in (
            "epoch",
            "train_loss",
            "val_loss",
            "val_auroc",
            "val_auprc",
            "val_f1",
            "val_f1_tuned",
            "val_threshold",
            "epoch_time",
        )
    }
    best_score, best_epoch, best_threshold_, best_state, stale = (
        float("-inf"),
        0,
        0.5,
        None,
        0,
    )
    fit_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        start, loss_sum, n = time.time(), 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            batch = to_device(batch)
            logits = forward(model, batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["labels"], pos_weight=pw
            )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(
                opt
            )  # so that the clipping threshold is in real gradient units
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * len(batch["labels"])
            n += len(batch["labels"])
        epoch_time = time.time() - start  # training only, validation excluded

        val_logits, val_labels, _ = predict(model, val_loader)
        at_half = compute_metrics(val_logits, val_labels)  # threshold 0.5
        threshold = best_threshold(val_logits, val_labels)
        tuned = compute_metrics(val_logits, val_labels, threshold)

        for key, value in zip(
            history,
            (
                epoch,
                loss_sum / n,
                at_half["loss"],
                at_half["auroc"],
                at_half["auprc"],
                at_half["f1"],
                tuned["f1"],
                threshold,
                epoch_time,
            ),
        ):
            history[key].append(value)
        tqdm.write(
            f"epoch {epoch:2d}/{epochs} | train loss {loss_sum / n:.4f} | val loss {at_half['loss']:.4f} "
            f"| val AUROC {at_half['auroc']:.4f} | val AUPRC {at_half['auprc']:.4f} "
            f"| val F1 {at_half['f1']:.4f} "
            f"| val F1 tuned {tuned['f1']:.4f} (thr {threshold:.2f}) | {epoch_time:.0f} s"
        )

        # Higher is better for every monitored score, so the loss is negated.
        score = {
            "auprc": at_half["auprc"],
            "auroc": at_half["auroc"],
            "f1_tuned": tuned["f1"],
            "loss": -at_half["loss"],
        }[monitor]

        if score > best_score + min_delta:
            best_score, best_epoch, best_threshold_, stale = (
                score,
                epoch,
                threshold,
                0,
            )
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                tqdm.write(
                    f"early stopping: no val {monitor} improvement for {patience} epochs"
                )
                break

    # Restore the best epoch.
    model.load_state_dict(best_state)

    history["monitor"] = monitor
    history["best_epoch"] = best_epoch
    history["best_threshold"] = best_threshold_
    history["total_time"] = time.time() - fit_start

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return history
