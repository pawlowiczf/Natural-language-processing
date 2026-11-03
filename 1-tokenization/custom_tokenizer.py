import json
import re
from collections import Counter

# A token is either a run of letters/digits or a single punctuation mark
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
UNK = "<UNK>"


class WhitespaceTokenizer:
    def __init__(self, vocab_size=50257, lowercase=True):
        # vocab_size includes the <UNK> token; use the same N as in the BPE tokenizer
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        # Split text into words and separate punctuation tokens
        if self.lowercase:
            text = text.lower()
        return TOKEN_PATTERN.findall(text)

    def train(self, corpus):
        # Count token frequencies over the whole corpus
        counter = Counter()
        for text in corpus:
            counter.update(self.tokenize(text))

        # Reserve id 0 for <UNK>, fill the rest with the most frequent tokens
        self.token_to_id = {UNK: 0}
        for token, _ in counter.most_common(self.vocab_size - 1):
            self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def save(self, path):
        # Persist the trained vocabulary so training does not have to be repeated
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "lowercase": self.lowercase,
                    "token_to_id": self.token_to_id,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"], lowercase=data["lowercase"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {i: t for t, i in tokenizer.token_to_id.items()}
        return tokenizer

    def encode(self, text):
        # Out-of-vocabulary tokens are mapped to the <UNK> id
        unk_id = self.token_to_id[UNK]
        return [self.token_to_id.get(t, unk_id) for t in self.tokenize(text)]

    def decode(self, ids):
        # Join tokens with spaces and remove the space before punctuation
        text = " ".join(self.id_to_token[i] for i in ids)
        return re.sub(r"\s+([,.!?;:])", r"\1", text)


if __name__ == "__main__":
    corpus = [
        "Hello, world! How are you?",
        "Tokenizers split text into words, and punctuation.",
        "Hello again, world.",
    ]

    # Small N so that out-of-vocabulary words are visible in the demo
    tok = WhitespaceTokenizer(vocab_size=12)
    tok.train(corpus)

    text = "Hello, brave new world!"
    print(tok.tokenize(text))  # ['hello', ',', 'brave', 'new', 'world', '!']
    ids = tok.encode(text)
    print(ids)  # 'brave' and 'new' become id 0 (<UNK>)
    print(tok.decode(ids))  # hello, <UNK> <UNK> world!
