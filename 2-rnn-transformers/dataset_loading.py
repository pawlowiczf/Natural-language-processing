"""Shared data pipeline for every architecture notebook (RNN, Transformer, ...).

    from dataset_loading import prepare, get_batch, evaluate
    tok = prepare()                                   # once; later calls hit the cache
    x, y = get_batch("train", batch_size, block_size, device)
    val_loss, val_ppl = evaluate(model, "val", block_size, batch_size, device)

Splits (all tokenized with the same tokenizer, stored as uint16 .bin files):
    train - fineweb-edu, everything except the last VAL_TOKENS
    val   - the last VAL_TOKENS of the fineweb-edu stream (same domain as train)
    test  - Wikipedia (wikitext-103-raw-v1 test split), held-out, different domain

The model contract is simply: model(x[B, T] int64) -> logits[B, T, vocab].
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA_DIR = Path(__file__).parent / "data"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
EOT = "<|endoftext|>"
VOCAB_SIZE = 16384  # must stay below 65536 for uint16 storage

TOKENIZER_DOCS = 200_000
TRAIN_TOKENS = 500_000_000  # fineweb-edu tokens to pull (train + val); lower for quick runs
VAL_TOKENS = 10_000_000  # tail of the fineweb file held out for validation
_BATCH_DOCS = 1000

_PATHS = {"train": DATA_DIR / "fineweb.bin", "val": DATA_DIR / "fineweb.bin", "test": DATA_DIR / "wiki_test.bin"}


def _fineweb_texts(n_docs=None):
    # Streaming: nothing is downloaded in full
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if n_docs is not None and i >= n_docs:
            return
        yield ex["text"]


def _wiki_texts():
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    # rows are lines that already end in "\n"; join them back into one document
    yield "".join(wiki["text"])


def get_tokenizer() -> Tokenizer:
    """Byte-level BPE, shared by all models. Trained once, then loaded from disk."""
    if TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    DATA_DIR.mkdir(exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_fineweb_texts(TOKENIZER_DOCS), trainer)
    tok.save(str(TOKENIZER_PATH))
    return tok


def _write_bin(path, tok, texts, max_tokens=None):
    """Tokenize an iterable of documents (EOT after each) into a flat uint16 file."""
    eot = tok.token_to_id(EOT)
    n = 0
    with open(path, "wb") as f:
        batch = []
        for text in texts:
            batch.append(text)
            if len(batch) < _BATCH_DOCS:
                continue
            n += _flush(f, tok, batch, eot)
            batch = []
            if max_tokens and n >= max_tokens:
                return
        if batch:
            _flush(f, tok, batch, eot)


def _flush(f, tok, batch, eot):
    ids = [i for enc in tok.encode_batch(batch) for i in (*enc.ids, eot)]
    np.asarray(ids, dtype=np.uint16).tofile(f)
    return len(ids)


def prepare() -> Tokenizer:
    """Build tokenizer and all .bin files if missing. Safe to call from every notebook."""
    tok = get_tokenizer()
    if not _PATHS["train"].exists():
        tmp = _PATHS["train"].with_suffix(".tmp")  # no half-written file if interrupted
        _write_bin(tmp, tok, _fineweb_texts(), TRAIN_TOKENS)
        tmp.rename(_PATHS["train"])
    if not _PATHS["test"].exists():
        _write_bin(_PATHS["test"], tok, _wiki_texts())
    return tok


def _data(split):
    arr = np.memmap(_PATHS[split], dtype=np.uint16, mode="r")
    if split == "train":
        return arr[:-VAL_TOKENS]
    if split == "val":
        return arr[-VAL_TOKENS:]
    return arr


def get_batch(split, batch_size, block_size, device):
    """Random windows; y is x shifted by one (next-token targets)."""
    data = _data(split)
    ix = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


if __name__ == "__main__":
    prepare()  # python dataset_loading.py: builds tokenizer + all .bin files
