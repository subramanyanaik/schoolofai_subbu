"""Vocabulary packing.

The one design choice that makes everything downstream cheap: tokens are sorted
ascending by byte length.  After that, ``{v : L_v > p}`` is a *contiguous
suffix* for every position p, which is what lets the loop implementation be a
slice-add and lets the occupancy matrix be built without any padding.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[^\s]+")


def gpt2_token_bytes() -> list[bytes]:
    """The 50,257 GPT-2 surface forms, as raw bytes.

    A handful of ids in some encodings are reserved and have no surface form;
    those fall back to a single NUL byte so that every row of the occupancy
    matrix has at least one entry (an empty bag would be a silent zero row).
    """
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    out: list[bytes] = []
    for i in range(enc.n_vocab):
        try:
            b = enc.decode_single_token_bytes(i)
        except Exception:
            b = b"\x00"
        out.append(b if b else b"\x00")
    return out


class PackedVocab:
    """Length-sorted vocabulary plus the flat arrays the head needs.

    The sort key is the *true* byte length, which is independent of ``L_max``.
    That is deliberate: ``order``/``inv_order`` stay identical across every
    codec configuration, so a tokenized data file cached at one L_max is valid
    at every other one (E5 sweeps L_max and must not re-tokenize).
    """

    def __init__(self, token_bytes, L_max: int) -> None:
        token_bytes = [bytes(t) if t else b"\x00" for t in token_bytes]
        self.L_max = int(L_max)
        true_lengths = np.array([len(t) for t in token_bytes], dtype=np.int64)

        # Stable sort => ties keep original id order => fully deterministic.
        self.order = np.argsort(true_lengths, kind="stable").astype(np.int64)
        self.inv_order = np.empty_like(self.order)
        self.inv_order[self.order] = np.arange(self.order.size, dtype=np.int64)

        self.token_bytes = [token_bytes[i] for i in self.order]
        self.true_lengths = true_lengths[self.order]
        self.lengths = np.minimum(self.true_lengths, self.L_max)
        self._sorted_bytes = [t[: self.L_max] for t in self.token_bytes]

        self.V = int(self.order.size)
        self.max_len = int(self.lengths.max())
        self.total_bytes = int(self.lengths.sum())  # B_V
        self.n_truncated = int((self.true_lengths > self.L_max).sum())

        # hi[p] = first packed index whose clipped length exceeds p.
        self.hi = np.searchsorted(self.lengths, np.arange(self.L_max + 1), side="right")

        self.flat_bytes = np.frombuffer(b"".join(self._sorted_bytes), dtype=np.uint8)
        self.offsets = np.concatenate(
            [[0], np.cumsum(self.lengths)[:-1]]
        ).astype(np.int64)
        self.inv_sqrt_len = (1.0 / np.sqrt(self.lengths)).astype(np.float64)
        self.pos_mod = (
            np.arange(self.total_bytes, dtype=np.int64)
            - np.repeat(self.offsets, self.lengths)
        ) % self.L_max

        assert np.all(np.diff(self.lengths) >= 0), "packing must be length-sorted"
        assert self.flat_bytes.size == self.total_bytes

    def __len__(self) -> int:
        return self.V

    def bytes_at(self, p: int) -> np.ndarray:
        """Bytes at position p for the contiguous suffix that reaches it."""
        lo = int(self.hi[p])
        return np.array([t[p] for t in self._sorted_bytes[lo:]], dtype=np.int64)

    def to_packed(self, ids) -> np.ndarray:
        return self.inv_order[np.asarray(ids)]

    def to_original(self, ids) -> np.ndarray:
        return self.order[np.asarray(ids)]

    def summary(self) -> dict:
        return {
            "V": self.V,
            "L_max": self.L_max,
            "max_len": self.max_len,
            "mean_len": float(self.true_lengths.mean()),
            "total_bytes_B_V": self.total_bytes,
            "n_truncated": self.n_truncated,
            "frac_truncated": self.n_truncated / self.V,
        }


def build_large_vocab(
    target: int,
    base: list[bytes],
    texts,
    L_max: int = 128,
    seed: int = 0,
) -> list[bytes]:
    """Grow ``base`` to ``target`` surface forms for the E6 vocabulary swap.

    Real word forms are harvested from ``texts`` first (frequency-ordered, with
    a leading-space variant, which is how BPE vocabularies actually look).  If
    the corpus does not supply enough distinct forms, the remainder is filled
    with deterministic synthetic byte strings -- E6 is a feasibility and
    parameter-count experiment, and the filler is disclosed rather than hidden.
    """
    seen = set(base)
    out = list(base)
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_WORD_RE.findall(text))

    harvested = 0
    for word, _ in counts.most_common():
        for form in (word.encode("utf-8"), (" " + word).encode("utf-8")):
            if len(out) >= target:
                break
            if form and form not in seen and len(form) <= L_max:
                seen.add(form)
                out.append(form)
                harvested += 1
        if len(out) >= target:
            break

    rng = np.random.default_rng(seed)
    n_synthetic = 0
    while len(out) < target:
        n = int(rng.integers(3, 13))
        form = bytes(rng.integers(33, 127, size=n, dtype=np.uint8).tolist())
        if form not in seen:
            seen.add(form)
            out.append(form)
            n_synthetic += 1

    build_large_vocab.last_stats = {  # type: ignore[attr-defined]
        "base": len(base),
        "harvested": harvested,
        "synthetic": n_synthetic,
        "total": len(out),
    }
    return out[:target]
