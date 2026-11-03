# File was created with help of AI, due to the complexity of model creation.

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TrainConfig:
    # Everything except the tokenizer must be identical across the three experiments
    vocab_size: int = 50257
    block_size: int = 256  # context length in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    batch_size: int = 16
    micro_batch_size: int = 4  # batch_size is split into chunks of this size (gradient accumulation) to save GPU memory
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


class TransformerLM(nn.Module):
    # Decoder-style language model built from PyTorch's stock nn.TransformerEncoder
    # (self-attention only) with a causal mask; input/output embeddings are tied
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.n_layer, enable_nested_tensor=False
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.ln_f(h))


def lm_loss(model, x):
    # Next-token cross-entropy: position t predicts token t + 1
    logits = model(x)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1)
    )


def build_model(config):
    # Randomly initialised model, no pretrained weights
    return TransformerLM(config)


def get_batch(ids, config, generator, device):
    # Random windows of block_size tokens. Starts are drawn as fractions of the corpus, so every
    # tokenizer sees the same relative positions even though the corpora differ in length
    fractions = torch.rand(config.batch_size, generator=generator)
    starts = (fractions * (len(ids) - config.block_size)).long()
    x = torch.stack(
        [
            torch.from_numpy(ids[s : s + config.block_size].astype(np.int64))
            for s in starts
        ]
    )
    return x.to(device)


def get_lr(step, config):
    # Linear warmup followed by cosine decay to 10% of the peak learning rate
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    return config.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, ids, config, device, n_chars):
    # Cross-entropy over consecutive non-overlapping blocks of the validation ids.
    # Returns (nats per token, bits per character). Bits per character is the only number
    # comparable across tokenizers, because they split the same text into different numbers of tokens.
    model.eval()
    n_blocks = (len(ids) - 1) // config.block_size
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n_blocks, config.micro_batch_size):
        blocks = range(start, min(start + config.micro_batch_size, n_blocks))
        x = torch.stack(
            [
                torch.from_numpy(
                    ids[b * config.block_size : (b + 1) * config.block_size].astype(
                        np.int64
                    )
                )
                for b in blocks
            ]
        ).to(device)
        loss = lm_loss(model, x)  # mean over block_size - 1 predictions per block
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
    generator = torch.Generator().manual_seed(
        config.seed
    )  # same relative batch positions for every run

    model = build_model(config).to(device)
    print(
        f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {device}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    run_dir = os.path.join(config.out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    log = []
    start_time = time.time()

    for step in range(config.max_steps + 1):
        if step % config.eval_every == 0:
            nats, bpc = evaluate(model, val_ids, config, device, val_chars)
            log.append(
                {"step": step, "val_nats_per_token": nats, "val_bits_per_char": bpc}
            )
            print(
                f"step {step}: val loss {nats:.3f} nats/token, {bpc:.3f} bits/char, {time.time() - start_time:.0f}s"
            )
        if step == config.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, config)
        x = get_batch(train_ids, config, generator, device)
        # Gradient accumulation: same effective batch, but logits are only built for a few sequences at once
        for chunk in x.split(config.micro_batch_size):
            loss = lm_loss(model, chunk) * chunk.shape[0] / x.shape[0]
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump({"config": asdict(config), "log": log}, f, indent=2)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    return log
