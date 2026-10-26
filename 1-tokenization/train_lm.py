import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass
class TrainConfig:
    # Everything except the tokenizer must be identical across the three experiments
    vocab_size: int = 50257
    block_size: int = 256  # context length in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    batch_size: int = 16
    max_steps: int = 2000
    lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    eval_every: int = 250
    seed: int = 42
    out_dir: str = "runs"


def encode_corpus(encode, texts, path):
    # Encode texts into one long id array and cache it on disk (ids fit in uint16 for N < 65536)
    if os.path.exists(path):
        return np.load(path)
    ids = []
    for text in texts:
        ids.extend(encode(text))
    ids = np.array(ids, dtype=np.uint16)
    np.save(path, ids)
    return ids


def build_model(config):
    # Randomly initialised GPT-2 (no pretrained weights); input/output embeddings are tied
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.block_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
    )
    return GPT2LMHeadModel(gpt2_config)


def get_batch(ids, config, generator, device):
    # Random windows of block_size tokens; labels are the same window (HF shifts them internally)
    starts = torch.randint(len(ids) - config.block_size, (config.batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(ids[s : s + config.block_size].astype(np.int64)) for s in starts])
    return x.to(device)


def get_lr(step, config):
    # Linear warmup followed by cosine decay to 10% of the peak learning rate
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    return config.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, ids, config, device, n_chars):
    # Cross-entropy over consecutive non-overlapping blocks of the validation ids.
    # Returns (nats per token, bits per character). Bits per character is the only number
    # comparable across tokenizers, because they split the same text into different numbers of tokens.
    model.eval()
    n_blocks = (len(ids) - 1) // config.block_size
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n_blocks, config.batch_size):
        blocks = range(start, min(start + config.batch_size, n_blocks))
        x = torch.stack(
            [torch.from_numpy(ids[b * config.block_size : (b + 1) * config.block_size].astype(np.int64)) for b in blocks]
        ).to(device)
        loss = model(input_ids=x, labels=x).loss  # mean over block_size - 1 predictions per block
        n_predicted = x.shape[0] * (config.block_size - 1)
        total_loss += loss.item() * n_predicted
        total_tokens += n_predicted
    model.train()
    nats_per_token = total_loss / total_tokens
    # Scale the character count to the share of tokens actually evaluated
    chars_covered = n_chars * total_tokens / len(ids)
    bits_per_char = nats_per_token * total_tokens / (chars_covered * math.log(2))
    return nats_per_token, bits_per_char


def train(name, train_ids, val_ids, val_chars, config):
    # Train a fresh model on pre-encoded ids; saves the log and weights to <out_dir>/<name>
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)  # same batch positions for every run

    model = build_model(config).to(device)
    print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    run_dir = os.path.join(config.out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    log = []
    start_time = time.time()

    for step in range(config.max_steps + 1):
        if step % config.eval_every == 0:
            nats, bpc = evaluate(model, val_ids, config, device, val_chars)
            log.append({"step": step, "val_nats_per_token": nats, "val_bits_per_char": bpc})
            print(f"step {step}: val loss {nats:.3f} nats/token, {bpc:.3f} bits/char, {time.time() - start_time:.0f}s")
        if step == config.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, config)
        x = get_batch(train_ids, config, generator, device)
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump({"config": asdict(config), "log": log}, f, indent=2)
    model.save_pretrained(run_dir)
    return log
