"""
tokenizer.py — a frozen, content-hashed byte-level BPE.

The tokenizer itself is a proxy: 512 entries, trained on the toy corpus,
standing in for the Session-3 tokenizer (vocab 196,608). What is NOT a proxy is
the contract around it, because that is what Session 6 is graded on:

  * It is trained ONCE, by `tools/build_tokenizer.py`, and the result is
    committed. Nothing in the demo trains a tokenizer; the demo LOADS one.
  * Its serialized form has exactly one canonical encoding, so its hash is
    reproducible by anyone holding the file.
  * That hash is stamped into every shard manifest. A shard whose recorded
    tokenizer hash does not match the loaded tokenizer cannot be admitted --
    token ids without their tokenizer are meaningless integers.

Byte-level was chosen deliberately over a word-level toy: Devanagari and Tamil
are multi-byte in UTF-8, so a byte-level vocabulary exercises the same
merge-across-byte-boundary behaviour the real Indic tokenizer has to handle, and
guarantees no document is unencodable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from . import config as C
from .hashing import hash_obj

# Layout of the id space. Fixed by construction and part of the hash.
N_SPECIAL = len(C.SPECIAL_TOKENS)
BYTE_BASE = N_SPECIAL              # byte b -> id BYTE_BASE + b
MERGE_BASE = BYTE_BASE + 256       # first learned merge id


class Tokenizer:
    """A loaded, frozen tokenizer. Immutable once constructed."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.name: str = spec["name"]
        self.version: str = spec["version"]
        self.special_tokens: List[str] = list(spec["special_tokens"])
        # merges[i] = (left_id, right_id) producing id MERGE_BASE + i
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in spec["merges"]]
        self.vocab_size: int = MERGE_BASE + len(self.merges)

        self._rank: Dict[Tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        # id -> raw bytes, for decoding and for token previews in the trace
        self._bytes: List[bytes] = [b""] * self.vocab_size
        for i, tok in enumerate(self.special_tokens):
            self._bytes[i] = tok.encode("utf-8")
        for b in range(256):
            self._bytes[BYTE_BASE + b] = bytes([b])
        for i, (l, r) in enumerate(self.merges):
            self._bytes[MERGE_BASE + i] = self._bytes[l] + self._bytes[r]

        self.hash: str = tokenizer_hash(spec)

    # -- encoding ----------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Encode raw text. Special tokens are NOT parsed out of text; role
        markers are inserted structurally by the packer, never by string
        matching, so a document cannot smuggle in a control token."""
        ids = [BYTE_BASE + b for b in text.encode("utf-8")]
        if len(ids) < 2:
            return ids
        return self._merge(ids)

    def _merge(self, ids: List[int]) -> List[int]:
        while True:
            best_rank = None
            best_at = -1
            for i in range(len(ids) - 1):
                rank = self._rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_at = rank, i
            if best_at < 0:
                return ids
            new_id = MERGE_BASE + best_rank
            ids = ids[:best_at] + [new_id] + ids[best_at + 2:]
            if len(ids) < 2:
                return ids

    # -- decoding ----------------------------------------------------------
    def decode(self, ids: Sequence[int]) -> str:
        raw = b"".join(self._bytes[i] for i in ids if 0 <= i < self.vocab_size)
        return raw.decode("utf-8", errors="replace")

    def preview(self, token_id: int) -> str:
        """Human-readable single-token preview for the token-level trace."""
        if token_id < N_SPECIAL:
            return self.special_tokens[token_id]
        raw = self._bytes[token_id]
        text = raw.decode("utf-8", errors="replace")
        return text.replace("\n", "\\n").replace("\t", "\\t")

    def is_special(self, token_id: int) -> bool:
        return token_id < N_SPECIAL


def canonical_spec(name: str, version: str, special_tokens: Sequence[str],
                   merges: Sequence[Tuple[int, int]]) -> dict:
    """The one serialized form whose hash is the tokenizer identity."""
    return {
        "name": name,
        "version": version,
        "byte_level": True,
        "n_special": len(special_tokens),
        "special_tokens": list(special_tokens),
        "byte_base": BYTE_BASE,
        "merge_base": MERGE_BASE,
        "merges": [[int(a), int(b)] for a, b in merges],
        "vocab_size": MERGE_BASE + len(merges),
    }


def tokenizer_hash(spec: dict) -> str:
    return hash_obj(spec)


def save(spec: dict, path: Path | str = None) -> Path:
    path = Path(path or C.TOKENIZER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load(path: Path | str = None) -> Tokenizer:
    path = Path(path or C.TOKENIZER_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"frozen tokenizer missing at {path}. Run tools/build_tokenizer.py once; "
            "the demo loads a tokenizer, it never trains one."
        )
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    return Tokenizer(spec)


# ---------------------------------------------------------------------------
# Training. Used only by tools/build_tokenizer.py.
# ---------------------------------------------------------------------------
def train(texts: Iterable[str], vocab_size: int = None,
          special_tokens: Sequence[str] = None) -> dict:
    """Learn byte-pair merges until the vocabulary reaches `vocab_size`.

    Deterministic: pair frequencies are counted over the corpus in a fixed
    order and ties are broken by the (left, right) id pair, so the same corpus
    always yields the same merge list and therefore the same tokenizer hash.
    """
    vocab_size = vocab_size or C.VOCAB_SIZE
    special_tokens = list(special_tokens or C.SPECIAL_TOKENS)
    n_merges = vocab_size - (len(special_tokens) + 256)
    if n_merges < 0:
        raise ValueError("vocab_size smaller than the byte + special floor")

    corpus: List[List[int]] = [
        [BYTE_BASE + b for b in t.encode("utf-8")] for t in texts if t
    ]
    merges: List[Tuple[int, int]] = []

    for step in range(n_merges):
        counts: Dict[Tuple[int, int], int] = {}
        for seq in corpus:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        # Highest count wins; ties broken by pair value for determinism.
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if counts[best] < 2:
            break
        new_id = MERGE_BASE + step
        merges.append(best)
        a, b = best
        for si, seq in enumerate(corpus):
            if len(seq) < 2:
                continue
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            corpus[si] = out

    return canonical_spec(C.TOKENIZER_NAME, C.TOKENIZER_VERSION,
                          special_tokens, merges)
