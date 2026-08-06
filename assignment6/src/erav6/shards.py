"""
shards.py — documents become immutable tokenized objects.

A shard is a sealed training object: a flat little-endian uint32 array of token
ids (`<id>.bin`) plus a side index (`<id>.idx.json`) that says which document
each region came from and what role every token plays. Once written, a shard is
never edited. Editing it produces a different content hash, and the admission
gate refuses it -- which is the property `demo` exercises by tampering with a
copy and watching it get rejected.

The index is the reason this system can replay. A ledger event stores token
SPANS (`shard_id:start:length`), not token values. To reconstruct a batch months
later you re-open the shard, cut the recorded spans, and rebuild. That only
works if the shard is genuinely immutable and the index is exact, so both are
hashed.

Role spans are laid down here rather than at packing time. Every token in a
document belongs to exactly one contiguous span, and each span carries its
loss flag from `config.ROLE_LOSS_POLICY`. That is where "tool observations are
context-only" stops being a design note and becomes a bit in a file.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from . import config as C
from .hashing import canonical_json, hash_file, sha256_text
from .tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def load_documents(path: Path | str = None) -> List[dict]:
    path = Path(path or C.CORPUS_DIR / "documents.jsonl")
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


@dataclass
class TokenizedDoc:
    doc_id: str
    lane: str
    split: str
    language: str
    script: str
    source_id: str
    meta: dict
    tokens: List[int]
    # Contiguous, exhaustive cover of `tokens`. Every token is in exactly one.
    spans: List[dict] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def loss_mask(self) -> List[int]:
        mask = [0] * len(self.tokens)
        for sp in self.spans:
            if sp["loss"]:
                for i in range(sp["start"], sp["start"] + sp["length"]):
                    mask[i] = 1
        return mask

    @property
    def n_loss_tokens(self) -> int:
        return sum(sp["length"] for sp in self.spans if sp["loss"])


def tokenize_document(doc: dict, tok: Tokenizer) -> TokenizedDoc:
    """Lay out one document as tokens + an exhaustive span cover.

    Layout:  <bos> [<role>] segment... [<role>] segment... <eos>

    Role markers are emitted structurally, from the document's declared role,
    never by matching control strings inside the text. A document therefore
    cannot smuggle in an `<|assistant|>` marker to make its own text
    loss-bearing.
    """
    tokens: List[int] = [C.BOS_ID]
    spans: List[dict] = [dict(role="bos", start=0, length=1, loss=0)]

    last_loss = 1
    for seg in doc["segments"]:
        role = seg["role"]
        if role not in C.ROLE_LOSS_POLICY:
            raise ValueError(f"{doc['doc_id']}: unknown role {role!r}")
        loss = 1 if C.ROLE_LOSS_POLICY[role] else 0

        # The role marker itself is structural, never loss-bearing: we do not
        # want the model spending capacity predicting scaffolding.
        if role in C.ROLE_TOKEN_IDS:
            spans.append(dict(role=f"marker:{role}", start=len(tokens),
                              length=1, loss=0))
            tokens.append(C.ROLE_TOKEN_IDS[role])

        ids = tok.encode(seg["text"])
        if ids:
            spans.append(dict(role=role, start=len(tokens), length=len(ids), loss=loss))
            tokens.extend(ids)
            last_loss = loss

    # EOS inherits the last content segment's policy. For plain pretraining that
    # means the model learns where documents end; for a trajectory ending in a
    # context-only span it stays masked.
    spans.append(dict(role="eos", start=len(tokens), length=1, loss=last_loss))
    tokens.append(C.EOS_ID)

    _assert_span_cover(spans, len(tokens), doc["doc_id"])

    return TokenizedDoc(
        doc_id=doc["doc_id"], lane=doc["lane"], split=doc["split"],
        language=doc["language"], script=doc["script"],
        source_id=doc["source_id"], meta=doc["meta"],
        tokens=tokens, spans=spans,
    )


def _assert_span_cover(spans: Sequence[dict], n: int, doc_id: str) -> None:
    """Spans must tile the document exactly: no gap, no overlap.

    A gap would be a token with no declared role and therefore no defined loss
    policy -- exactly the kind of silent hole this system exists to prevent.
    """
    cursor = 0
    for sp in spans:
        if sp["start"] != cursor:
            raise AssertionError(f"{doc_id}: span gap/overlap at {sp['start']} (expected {cursor})")
        cursor += sp["length"]
    if cursor != n:
        raise AssertionError(f"{doc_id}: spans cover {cursor} of {n} tokens")


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------
@dataclass
class Shard:
    shard_id: str
    lane: str
    split: str
    token_count: int
    docs: List[dict]                  # index records
    bin_path: Path
    idx_path: Path

    _cache: Optional[List[int]] = field(default=None, repr=False, compare=False)

    # -- reading -----------------------------------------------------------
    def tokens(self) -> List[int]:
        """Whole-shard token array. Cached; `perf` measures the cache."""
        if self._cache is None:
            raw = self.bin_path.read_bytes()
            self._cache = list(struct.unpack(f"<{len(raw)//4}I", raw))
        return self._cache

    def read_span(self, start: int, length: int) -> List[int]:
        toks = self.tokens()
        if start < 0 or start + length > len(toks):
            raise IndexError(f"{self.shard_id}: span {start}+{length} out of range")
        return toks[start:start + length]

    def doc(self, doc_id: str) -> dict:
        for d in self.docs:
            if d["doc_id"] == doc_id:
                return d
        raise KeyError(f"{self.shard_id}: no document {doc_id}")

    def index_struct(self) -> dict:
        return dict(shard_id=self.shard_id, lane=self.lane, split=self.split,
                    token_count=self.token_count, docs=self.docs)


# Documents are grouped until a shard passes this. Deliberately small: a lane
# with several shards is a lane whose anneal reserve can hold one back, and
# whose learning ledger has more than one row to compare. Session 6's own advice
# on shard sizing -- small files cost overhead, huge files cost recovery
# flexibility -- is about production scale, not a 25KB corpus.
SHARD_TARGET_TOKENS = 450


def build_shards(docs: List[TokenizedDoc], out_dir: Path | str = None,
                 tokenizer_hash: str = "") -> List[Shard]:
    """Group tokenized documents into shards.

    Shards are grouped by (lane, split, licence), and never straddle any of the
    three:

      * LANE, because the mixture scheduler selects at lane granularity. A
        mixed-lane shard could not be attributed to a lane quota, and the
        "planned versus actual share" evidence would be unverifiable.
      * SPLIT, because train and eval data have different permissions and a
        shard is admitted or refused as a whole.
      * LICENCE, because a shard's licence tier is the WORST licence among its
        members. Mixing an unclear-licence document into an otherwise permissive
        shard would take the whole shard down with it -- so real systems shard
        by licence tier, and so does this one.
    """
    out_dir = Path(out_dir or C.SHARD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[tuple, List[TokenizedDoc]] = {}
    for d in docs:
        groups.setdefault((d.lane, d.split, d.meta.get("license", "unknown")), []).append(d)

    shards: List[Shard] = []
    seq_by_lane_split: Dict[tuple, int] = {}

    for (lane, split, _license), members in sorted(groups.items()):
        members.sort(key=lambda d: d.doc_id)     # deterministic shard membership
        batch: List[TokenizedDoc] = []
        size = 0
        pending: List[List[TokenizedDoc]] = []
        for d in members:
            batch.append(d)
            size += d.n_tokens
            if size >= SHARD_TARGET_TOKENS:
                pending.append(batch)
                batch, size = [], 0
        if batch:
            pending.append(batch)

        for group in pending:
            key = (lane, split)
            seq = seq_by_lane_split.get(key, 0)
            seq_by_lane_split[key] = seq + 1
            shards.append(_write_shard(lane, split, seq, group, out_dir, tokenizer_hash))

    return shards


def _write_shard(lane: str, split: str, seq: int, members: List[TokenizedDoc],
                 out_dir: Path, tokenizer_hash: str) -> Shard:
    shard_id = f"shard-{lane}-{split}-{seq:03d}"
    tokens: List[int] = []
    index: List[dict] = []

    for d in members:
        start = len(tokens)
        tokens.extend(d.tokens)
        index.append(dict(
            doc_id=d.doc_id,
            start=start,
            length=d.n_tokens,
            split=d.split,
            language=d.language,
            script=d.script,
            source_id=d.source_id,
            license=d.meta.get("license"),
            n_loss_tokens=d.n_loss_tokens,
            # Span offsets are shard-relative, so a replay never needs the
            # document boundaries to reconstruct a mask.
            spans=[dict(role=sp["role"], start=start + sp["start"],
                        length=sp["length"], loss=sp["loss"]) for sp in d.spans],
        ))

    bin_path = out_dir / f"{shard_id}.bin"
    idx_path = out_dir / f"{shard_id}.idx.json"
    bin_path.write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))

    shard = Shard(shard_id=shard_id, lane=lane, split=split,
                  token_count=len(tokens), docs=index,
                  bin_path=bin_path, idx_path=idx_path)

    payload = shard.index_struct()
    payload["tokenizer_hash"] = tokenizer_hash
    with open(idx_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical_json(payload))
    return shard


def load_shard(shard_id: str, shard_dir: Path | str = None) -> Shard:
    shard_dir = Path(shard_dir or C.SHARD_DIR)
    idx_path = shard_dir / f"{shard_id}.idx.json"
    with open(idx_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return Shard(shard_id=payload["shard_id"], lane=payload["lane"],
                 split=payload["split"], token_count=payload["token_count"],
                 docs=payload["docs"],
                 bin_path=shard_dir / f"{shard_id}.bin", idx_path=idx_path)


def content_hash(shard: Shard) -> str:
    """Identity of a shard: its token bytes AND its index, together.

    Both halves matter. Same bytes with a shifted index is a different training
    object, because the spans decide which tokens carry loss.
    """
    return sha256_text(hash_file(shard.bin_path) + "\n" +
                       canonical_json(shard.index_struct()))


def verify(shard: Shard, expected_hash: str) -> bool:
    """Recompute the content hash from what is actually on disk."""
    return content_hash(shard) == expected_hash


def iter_doc_spans(shard: Shard) -> Iterator[dict]:
    for d in shard.docs:
        yield from d["spans"]
