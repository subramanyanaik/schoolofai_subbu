"""
build_tokenizer.py — trains the tokenizer ONCE and freezes it.

Run manually after any change to the corpus; the output
(`tokenizer/tokenizer.json`) is committed and is what the demo loads.

Freezing matters more than the vocabulary does. A training run whose tokenizer
can move underneath it has token ids that mean different things at different
points in the run, and every shard manifest, ledger event and replay claim
downstream becomes unverifiable. So the tokenizer is built here, hashed, and
the hash is printed for the record.

    python tools/build_tokenizer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

A6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A6 / "src"))

from erav6 import config as C          # noqa: E402
from erav6 import tokenizer as T       # noqa: E402


def corpus_texts():
    """Every segment of every document, TRAIN SPLIT ONLY.

    Training the tokenizer on held-out evaluation text would be a subtle form
    of contamination: the vocabulary would be fitted to benchmark strings.
    """
    path = C.CORPUS_DIR / "documents.jsonl"
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            if doc["split"] != "train":
                continue
            for seg in doc["segments"]:
                yield seg["text"]


def main():
    texts = list(corpus_texts())
    spec = T.train(texts, vocab_size=C.VOCAB_SIZE)
    path = T.save(spec)
    tok = T.load(path)

    sample = "मानसून monsoon calc(expr=\"3*45\") தமிழ்"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "round-trip failed"

    print(f"trained on {len(texts)} segments from the TRAIN split only")
    print(f"vocab_size      {tok.vocab_size}")
    print(f"merges learned  {len(tok.merges)}")
    print(f"tokenizer_hash  {tok.hash}")
    print(f"written to      {path}")
    print(f"round-trip      OK  ({len(sample)} chars -> {len(ids)} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
