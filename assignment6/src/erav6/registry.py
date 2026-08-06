"""
registry.py — the evaluation firewall.

"The system must know that evaluation data exists precisely so it can prevent
that data from entering a training batch."

Three permission levels sit in one registry:

    train       admitted into the gradient-bearing stream
    validation  READABLE during training for evaluation, never gradient-bearing
    test        never read by training at all; never_train is set

The distinction between validation and test is the one that is usually
implemented sloppily. Validation data is allowed into a forward pass, so the
protection it needs is not "never load this" but "never let this reach an
optimizer step". That is enforced structurally: `open_for_eval` stamps every
read with a purpose and refuses any purpose that is gradient-bearing, and the
loader has no code path that turns an eval read into a training batch.

Contamination detection is fingerprint-based rather than id-based, because the
interesting failure is not "somebody loaded the test shard". It is "a training
document happens to quote a benchmark item". Id checks cannot see that; shared
token n-grams can.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from . import config as C
from .hashing import sha256_bytes

GRADIENT_BEARING_PURPOSES = {"training", "gradient", "optimizer_step"}


def token_ngrams(tokens: Sequence[int], n: int = None) -> Set[str]:
    """Fingerprint set for a token array."""
    n = n or C.CONTAMINATION_NGRAM
    out: Set[str] = set()
    if len(tokens) < n:
        return out
    for i in range(len(tokens) - n + 1):
        buf = bytearray()
        for t in tokens[i:i + n]:
            buf += int(t).to_bytes(4, "little")
        out.add(sha256_bytes(bytes(buf))[:32])
    return out


@dataclass
class RegistryEntry:
    doc_id: str
    split: str                  # validation | test
    benchmark: str
    version: str
    never_train: bool
    content_hash: str
    fingerprints: Set[str] = field(default_factory=set, repr=False)


class EvalRegistry:
    """Registry of everything the training stream is not allowed to learn from."""

    def __init__(self):
        self.entries: Dict[str, RegistryEntry] = {}
        self._fingerprint_owner: Dict[str, str] = {}   # fingerprint -> doc_id
        self.access_log: List[dict] = []
        self.never_train_shards: Set[str] = set()

    # -- construction ------------------------------------------------------
    def register(self, doc_id: str, split: str, benchmark: str, version: str,
                 tokens: Sequence[int], never_train: bool) -> RegistryEntry:
        fps = token_ngrams(tokens)
        entry = RegistryEntry(
            doc_id=doc_id, split=split, benchmark=benchmark, version=version,
            never_train=never_train,
            content_hash=sha256_bytes(b"".join(int(t).to_bytes(4, "little") for t in tokens)),
            fingerprints=fps,
        )
        self.entries[doc_id] = entry
        for fp in fps:
            self._fingerprint_owner.setdefault(fp, doc_id)
        return entry

    def mark_never_train_shard(self, shard_id: str) -> None:
        self.never_train_shards.add(shard_id)

    # -- queries -----------------------------------------------------------
    @property
    def n_fingerprints(self) -> int:
        return len(self._fingerprint_owner)

    def benchmarks(self) -> List[str]:
        return sorted({e.benchmark for e in self.entries.values()})

    def scan_tokens(self, tokens: Sequence[int]) -> List[dict]:
        """Return every registry document this token array overlaps."""
        hits: Dict[str, int] = {}
        for fp in token_ngrams(tokens):
            owner = self._fingerprint_owner.get(fp)
            if owner:
                hits[owner] = hits.get(owner, 0) + 1
        out = []
        for doc_id, count in sorted(hits.items()):
            e = self.entries[doc_id]
            out.append(dict(eval_doc_id=doc_id, benchmark=e.benchmark,
                            version=e.version, split=e.split,
                            matched_ngrams=count, never_train=e.never_train))
        return out

    def scan_text(self, text: str) -> List[str]:
        """Canary check. Canaries exist to be found, so this is a plain match."""
        return [m for m in C.CANARY_MARKERS if m in text]

    def is_never_train_shard(self, shard_id: str) -> bool:
        return shard_id in self.never_train_shards

    # -- access control ----------------------------------------------------
    def open_for_eval(self, shard_id: str, purpose: str, step: int = -1) -> None:
        """Record a read of a registered shard, and refuse gradient purposes.

        Validation data is allowed through here. Test data is not, regardless of
        purpose -- a never-train shard has no legitimate read path inside the
        training process at all.
        """
        if purpose in GRADIENT_BEARING_PURPOSES:
            self.access_log.append(dict(shard_id=shard_id, purpose=purpose,
                                        step=step, allowed=False,
                                        reason="gradient_bearing_purpose_denied"))
            raise PermissionError(
                f"{shard_id}: refused read for gradient-bearing purpose {purpose!r}")
        if self.is_never_train_shard(shard_id):
            self.access_log.append(dict(shard_id=shard_id, purpose=purpose,
                                        step=step, allowed=False,
                                        reason="never_train_shard"))
            raise PermissionError(f"{shard_id}: never-train shard, no read path in training")
        self.access_log.append(dict(shard_id=shard_id, purpose=purpose,
                                    step=step, allowed=True, reason="validation_read"))

    # -- persistence -------------------------------------------------------
    def to_json(self) -> dict:
        return dict(
            n_entries=len(self.entries),
            n_fingerprints=self.n_fingerprints,
            ngram=C.CONTAMINATION_NGRAM,
            benchmarks=self.benchmarks(),
            never_train_shards=sorted(self.never_train_shards),
            entries=[
                dict(doc_id=e.doc_id, split=e.split, benchmark=e.benchmark,
                     version=e.version, never_train=e.never_train,
                     content_hash=e.content_hash, n_fingerprints=len(e.fingerprints))
                for e in sorted(self.entries.values(), key=lambda x: x.doc_id)
            ],
            access_log=self.access_log,
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_json(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        return path


def build_registry(tokenized_docs: Iterable) -> EvalRegistry:
    """Register every validation and test document."""
    reg = EvalRegistry()
    for d in tokenized_docs:
        if d.split == "train":
            continue
        reg.register(
            doc_id=d.doc_id,
            split=d.split,
            benchmark=d.meta.get("benchmark", "unknown"),
            version=d.meta.get("benchmark_version", "v0"),
            tokens=d.tokens,
            never_train=bool(d.meta.get("never_train", d.split == "test")),
        )
    return reg
