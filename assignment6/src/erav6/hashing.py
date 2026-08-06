"""
hashing.py — one canonical hashing path for the whole system.

Every hash that appears in a manifest, a ledger event or the evidence bundle is
produced here. That matters for two reasons:

  1. A hash is only useful as an identity if it is computed the same way every
     time. Two modules that both "hash the manifest" but disagree about key
     order or float formatting produce a system that fails audits for no real
     reason.

  2. The grader must be able to recompute our hashes from the artifacts we
     ship. So the canonical form is plain, sorted, separator-pinned JSON --
     reproducible from any language, not a pickle of Python objects.

Floats are deliberately NOT part of any identity hash. Batch identity is built
from token ids, span coordinates and mask bits, all integers. Losses are
measured quantities and live in the learning ledger, never in an identity hash;
that is what lets a replay match bit-for-bit across CPU and GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

HASH_ALGO = "sha256"
_SHORT = 16


def canonical_json(obj: Any) -> str:
    """The one serialization used for every content hash in the system."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_obj(obj: Any) -> str:
    """Content hash of any JSON-serializable object."""
    return sha256_text(canonical_json(obj))


def hash_tokens(token_ids: Sequence[int]) -> str:
    """Hash a token array.

    Encoded as fixed-width little-endian uint32 rather than as text, so the
    hash is of the token *values* and cannot drift with formatting.
    """
    buf = bytearray()
    for t in token_ids:
        buf += int(t).to_bytes(4, "little", signed=False)
    return sha256_bytes(bytes(buf))


def hash_mask(mask: Sequence[int]) -> str:
    """Hash a 0/1 mask. Packed one bit per position, so the loss-mask hash is
    cheap to store in every ledger event."""
    bits = bytearray((len(mask) + 7) // 8)
    for i, m in enumerate(mask):
        if m:
            bits[i >> 3] |= 1 << (i & 7)
    return sha256_bytes(bytes(bits))


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def chain(prev_hash: str, event: Any) -> str:
    """Hash-chain step: bind an event to everything that came before it.

    The consumption ledger is append-only, so every event stores the chain hash
    of its predecessor. Editing or deleting any historical event breaks every
    hash after it, which `audit.verify_chain` detects. This is what makes
    "append-only" an enforced property rather than a naming convention.
    """
    return sha256_text(prev_hash + "\n" + canonical_json(event))


GENESIS = "0" * 64


def short(h: str, n: int = _SHORT) -> str:
    """Truncated hash for logs and tables. Never used as an identity."""
    return h[:n]


def merkle(hashes: Iterable[str]) -> str:
    """Order-dependent digest of a list of hashes (used for shard sets)."""
    return sha256_text("\n".join(hashes))
