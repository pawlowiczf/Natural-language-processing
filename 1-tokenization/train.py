import argparse
import os
from dataclasses import fields

import sentencepiece as spm
from transformers import AutoTokenizer

from custom_tokenizer import WhitespaceTokenizer
from train_lm import TrainConfig, encode_corpus, train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_lines(path):
    # Documents are separated by a blank line; we train on single lines
    with open(path, encoding="utf-8") as f:
        docs = [d for d in f.read().split("\n\n") if d.strip()]
    return [line for doc in docs for line in doc.split("\n") if line.strip()]


def load_encoder(name, data_dir):
    # Tokenizers are trained in the notebook and loaded from data_dir
    if name == "bpe":
        return AutoTokenizer.from_pretrained("gpt2").encode
    if name == "whitespace":
        return WhitespaceTokenizer.load(os.path.join(data_dir, "ws_tokenizer.json")).encode
    return spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "sp.model")).encode


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, choices=["bpe", "whitespace", "sentencepiece"])
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, defaults.out_dir))
    parser.add_argument("--train-file", default=os.path.join(BASE_DIR, "plwiki_train.txt"))
    parser.add_argument("--eval-file", default=os.path.join(BASE_DIR, "plwiki_eval.txt"))
    parser.add_argument("--train-lines", type=int, default=200_000)
    for field in fields(TrainConfig):
        if field.name != "out_dir":
            parser.add_argument(
                "--" + field.name.replace("_", "-"), type=field.type, default=getattr(defaults, field.name)
            )
    args = parser.parse_args()

    config = TrainConfig(
        **{f.name: getattr(args, f.name) for f in fields(TrainConfig) if f.name != "out_dir"},
        out_dir=args.out_dir,
    )

    train_texts = read_lines(args.train_file)[: args.train_lines]
    val_texts = read_lines(args.eval_file)
    val_chars = sum(len(t) for t in val_texts)

    # Encoded ids are cached in data_dir, so only the first run with a tokenizer encodes the corpus
    encode = load_encoder(args.tokenizer, args.data_dir)
    train_ids = encode_corpus(encode, train_texts, os.path.join(args.data_dir, f"ids_{args.tokenizer}_train.npy"))
    val_ids = encode_corpus(encode, val_texts, os.path.join(args.data_dir, f"ids_{args.tokenizer}_val.npy"))
    print(f"{args.tokenizer}: {len(train_ids):,} train tokens, {len(val_ids):,} validation tokens")

    train(args.tokenizer, train_ids, val_ids, val_chars, config)


if __name__ == "__main__":
    main()
