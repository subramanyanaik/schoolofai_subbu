"""
packing.py — samples become fixed-shape windows, with their training meaning intact.

"The batch therefore needs to carry more than token IDs. It must also carry the
training meaning of those token IDs."

Every packed window emitted here carries five parallel arrays plus provenance:

    input_ids     what the model reads
    labels        what it is asked to predict (-100 where nothing is asked)
    loss_mask     1 exactly where a gradient is taken
    segment_ids   0 = padding, 1..k = mutually invisible samples
    position_ids  where each token believes it sits

The subtle one is `labels`. Loss at position i is a prediction of the token at
i+1, so the question "is this position loss-bearing" is really "is the NEXT
token loss-bearing, and is it legitimately predictable from here". Three things
can make it illegitimate: the next slot is padding, the next token starts a
different isolated sample, or the next token is context-only (a tool
observation). All three are handled in `_finalize`, in one place, for every
policy.

The policies differ in exactly one respect that matters -- whether samples
packed into the same window can see each other:

    concatenating policies   one segment; documents are separated by EOS and
                             the model may learn the EOS -> next-document
                             transition. Correct for plain pretraining.
    isolating policies       one segment per sample; attention is block
                             diagonal, position ids restart, and NO loss is
                             taken across a sample boundary. Required for SFT,
                             tool-use and reasoning traces, where an unrelated
                             neighbour in the context teaches transitions that
                             do not exist in the real task.

`loss_across_boundary` is reported on every window and must be zero for every
isolating policy. That is the packing-correctness invariant the test suite and
the evidence bundle both check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import config as C
from .hashing import hash_mask, hash_obj, hash_tokens

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Sample: a contiguous token span drawn from one immutable shard
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    shard_id: str
    doc_id: str
    lane: str
    start: int                      # absolute token offset within the shard
    length: int
    tokens: List[int]
    loss: List[int]                 # per-token loss flag, aligned with tokens
    spans: List[dict] = field(default_factory=list)   # role spans, shard-absolute

    @property
    def span_id(self) -> str:
        """The coordinate a ledger event stores. Values are never stored --
        a replay re-reads them from the shard using exactly this triple."""
        return f"{self.shard_id}:{self.start}:{self.length}"

    def role_boundaries(self) -> List[int]:
        """Sample-relative offsets at which a role span starts."""
        return [sp["start"] - self.start for sp in self.spans
                if 0 <= sp["start"] - self.start < self.length]


# ---------------------------------------------------------------------------
# Packed window
# ---------------------------------------------------------------------------
@dataclass
class PackedSequence:
    lane: str
    policy: str
    seq_len: int
    input_ids: List[int]
    labels: List[int]
    loss_mask: List[int]
    segment_ids: List[int]
    position_ids: List[int]
    members: List[dict] = field(default_factory=list)
    n_real_tokens: int = 0
    n_loss_tokens: int = 0
    n_truncated: int = 0
    boundary_crossings: int = 0
    loss_across_boundary: int = 0

    @property
    def utilization(self) -> float:
        return self.n_real_tokens / self.seq_len if self.seq_len else 0.0

    @property
    def loss_density(self) -> float:
        return self.n_loss_tokens / self.seq_len if self.seq_len else 0.0

    def loss_mask_hash(self) -> str:
        return hash_mask(self.loss_mask)

    def content_hash(self) -> str:
        """Identity of this window.

        Covers the tokens, the mask, the isolation structure and the position
        ids -- every array the optimizer actually sees. Two windows with the
        same tokens but different masks are different training objects and hash
        differently, which is what makes the replay check meaningful.
        """
        return hash_obj({
            "tokens": hash_tokens(self.input_ids),
            "loss_mask": hash_mask(self.loss_mask),
            "segment_ids": hash_tokens(self.segment_ids),
            "position_ids": hash_tokens(self.position_ids),
            "labels": hash_tokens([t if t >= 0 else 0 for t in self.labels]),
            "policy": self.policy,
            "lane": self.lane,
        })

    def provenance(self) -> List[str]:
        return [m["span_id"] for m in self.members]

    def stats(self) -> dict:
        return dict(lane=self.lane, policy=self.policy,
                    utilization=round(self.utilization, 6),
                    loss_density=round(self.loss_density, 6),
                    n_real_tokens=self.n_real_tokens,
                    n_loss_tokens=self.n_loss_tokens,
                    n_truncated=self.n_truncated,
                    n_segments=len({s for s in self.segment_ids if s}),
                    boundary_crossings=self.boundary_crossings,
                    loss_across_boundary=self.loss_across_boundary)


@dataclass
class PackResult:
    sequences: List[PackedSequence]
    n_samples_used: int
    policy: str

    def stats(self) -> dict:
        n = len(self.sequences) or 1
        return dict(
            policy=self.policy,
            n_sequences=len(self.sequences),
            n_samples_used=self.n_samples_used,
            mean_utilization=round(sum(s.utilization for s in self.sequences) / n, 6),
            mean_loss_density=round(sum(s.loss_density for s in self.sequences) / n, 6),
            total_real_tokens=sum(s.n_real_tokens for s in self.sequences),
            total_loss_tokens=sum(s.n_loss_tokens for s in self.sequences),
            total_truncated=sum(s.n_truncated for s in self.sequences),
            total_boundary_crossings=sum(s.boundary_crossings for s in self.sequences),
            total_loss_across_boundary=sum(s.loss_across_boundary for s in self.sequences),
        )


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------
class _Window:
    """Mutable builder for one packed sequence."""

    def __init__(self, seq_len: int, lane: str, policy: str, isolate: bool):
        self.seq_len = seq_len
        self.lane = lane
        self.policy = policy
        self.isolate = isolate
        self.ids = [C.PAD_ID] * seq_len
        self.loss = [0] * seq_len
        self.seg = [0] * seq_len
        self.pos = [0] * seq_len
        self.docidx = [-1] * seq_len        # which member owns each position
        self.cursor = 0
        self.n_segments = 0
        self.members: List[dict] = []
        self.n_truncated = 0

    @property
    def room(self) -> int:
        return self.seq_len - self.cursor

    def can_hold(self, n: int) -> bool:
        return self.room >= n

    def place(self, sample: Sample, limit: Optional[int] = None) -> int:
        """Write as much of `sample` as fits. Returns tokens written."""
        n = min(limit if limit is not None else sample.length, sample.length, self.room)
        if n <= 0:
            return 0

        offset = self.cursor
        member_idx = len(self.members)
        self.n_segments += 1
        seg_id = self.n_segments if self.isolate else 1

        for i in range(n):
            p = offset + i
            self.ids[p] = sample.tokens[i]
            self.loss[p] = sample.loss[i]
            self.seg[p] = seg_id
            # Isolating policies restart position ids per sample: a sample
            # packed second must not believe it begins at position 140, or the
            # model learns a positional offset that will not exist at inference.
            self.pos[p] = i if self.isolate else p
            self.docidx[p] = member_idx

        self.cursor += n
        truncated = sample.length - n
        self.n_truncated += truncated
        self.members.append(dict(
            span_id=f"{sample.shard_id}:{sample.start}:{n}",
            full_span_id=sample.span_id,
            shard_id=sample.shard_id,
            doc_id=sample.doc_id,
            lane=sample.lane,
            shard_start=sample.start,
            length=n,
            offset=offset,
            segment_id=seg_id,
            truncated=truncated,
        ))
        return n

    def place_continuation(self, sample: Sample, from_offset: int) -> int:
        """Concatenating policies may split a sample across windows."""
        remaining = sample.length - from_offset
        n = min(remaining, self.room)
        if n <= 0:
            return 0
        offset = self.cursor
        member_idx = len(self.members)
        self.n_segments += 1
        seg_id = 1                       # only concatenating policies call this
        for i in range(n):
            p = offset + i
            self.ids[p] = sample.tokens[from_offset + i]
            self.loss[p] = sample.loss[from_offset + i]
            self.seg[p] = seg_id
            self.pos[p] = p
            self.docidx[p] = member_idx
        self.cursor += n
        self.members.append(dict(
            span_id=f"{sample.shard_id}:{sample.start + from_offset}:{n}",
            full_span_id=sample.span_id,
            shard_id=sample.shard_id,
            doc_id=sample.doc_id,
            lane=sample.lane,
            shard_start=sample.start + from_offset,
            length=n,
            offset=offset,
            segment_id=seg_id,
            truncated=0,
            continuation=from_offset > 0,
        ))
        return n

    def finalize(self) -> PackedSequence:
        return _finalize(self)


def _finalize(w: _Window) -> PackedSequence:
    """Derive labels, loss mask and the boundary diagnostics.

    One rule, applied at every position i:

        take a gradient at i  <=>  i and i+1 are both real tokens
                                   AND (if isolating) they are the same segment
                                   AND token i+1 is itself loss-bearing
    """
    n = w.seq_len
    labels = [IGNORE_INDEX] * n
    lmask = [0] * n
    boundary = 0
    loss_across = 0

    for i in range(n - 1):
        if w.seg[i] == 0 or w.seg[i + 1] == 0:
            continue                                    # padding on either side
        crosses_doc = w.docidx[i] != w.docidx[i + 1]
        if crosses_doc:
            boundary += 1
        if w.isolate and w.seg[i] != w.seg[i + 1]:
            continue                                    # isolation boundary
        if not w.loss[i + 1]:
            continue                                    # context-only target
        labels[i] = w.ids[i + 1]
        lmask[i] = 1
        if crosses_doc:
            loss_across += 1

    return PackedSequence(
        lane=w.lane, policy=w.policy, seq_len=n,
        input_ids=list(w.ids), labels=labels, loss_mask=lmask,
        segment_ids=list(w.seg), position_ids=list(w.pos),
        members=list(w.members),
        n_real_tokens=w.cursor,
        n_loss_tokens=sum(lmask),
        n_truncated=w.n_truncated,
        boundary_crossings=boundary,
        loss_across_boundary=loss_across,
    )


# A span-boundary cut is only worth taking if it keeps most of the window. See
# `_truncate_at_span_boundary`.
MIN_BOUNDARY_CUT_FRACTION = 0.6


def _truncate_at_span_boundary(sample: Sample, limit: int) -> int:
    """Prefer to cut a structured sample where a role span ends -- but not at any cost.

    "Careless cuts can damage code blocks, proofs, agent trajectories and
    instruction-response examples where structure matters." Cutting a trajectory
    in the middle of a tool call leaves the model a call it can never see
    answered, so an isolating policy prefers to cut at a role boundary.

    Taken naively that rule backfires. A reasoning trace is
    `<user><think ...400 tokens...><assistant>`: its role boundaries sit at
    offsets 0, 1, 2, 40, 41 and 350, so the largest boundary under a 256-token
    window is 41 -- the end of the PROMPT. Cutting there yields a window
    containing only context-only tokens, with a loss density of exactly zero.
    Measured on this corpus before the guard: the reasoning lane produced
    windows with 1904 truncated tokens and no gradient at all.

    So a boundary cut is only taken when it retains at least
    MIN_BOUNDARY_CUT_FRACTION of the window. Otherwise the sample is cut hard
    at the limit, which keeps a damaged tail but keeps the loss-bearing content
    the window exists to train on. At Session-5 scale this case largely
    disappears, because these lanes get 32K-256K windows precisely so that a
    trace has room to finish.
    """
    if sample.length <= limit:
        return sample.length
    floor = int(MIN_BOUNDARY_CUT_FRACTION * limit)
    bounds = [b for b in sample.role_boundaries() if floor <= b <= limit]
    return max(bounds) if bounds else limit


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def pack(samples: Sequence[Sample], seq_len: int, policy: str,
         max_seqs: int, lane: str = "") -> PackResult:
    """Pack `samples` under `policy`, producing at most `max_seqs` windows.

    Only WHOLE samples are consumed (except in concatenating policies, which may
    legitimately split one across a window boundary and record both halves).
    `n_samples_used` tells the caller how far to advance its cursor; anything
    beyond that is handed back untouched, so no token is silently dropped.
    """
    if policy not in C.ALL_PACKING_POLICIES:
        raise ValueError(f"unknown packing policy {policy!r}")
    lane = lane or (samples[0].lane if samples else "")
    fn = _POLICIES[policy]
    return fn(list(samples), seq_len, max_seqs, lane, policy)


def _pack_pad_only(samples, seq_len, max_seqs, lane, policy):
    """One sample per window, right-padded. Maximum safety, minimum efficiency."""
    seqs, used = [], 0
    for s in samples:
        if len(seqs) >= max_seqs:
            break
        w = _Window(seq_len, lane, policy, isolate=True)
        w.place(s, limit=min(s.length, seq_len))
        seqs.append(w.finalize())
        used += 1
    return PackResult(seqs, used, policy)


def _pack_concat_and_chop(samples, seq_len, max_seqs, lane, policy):
    """Join the token streams and cut fixed windows.

    Efficient for plain pretraining and the only policy that treats a sequence
    boundary as a purely mechanical cut. Documents already carry EOS, so the
    boundary is still marked -- it is visible to the model, just not enforced by
    the attention mask.
    """
    seqs, used = [], 0
    w = _Window(seq_len, lane, policy, isolate=False)
    for s in samples:
        if len(seqs) >= max_seqs:
            break
        offset = 0
        while offset < s.length:
            if w.room == 0:
                seqs.append(w.finalize())
                if len(seqs) >= max_seqs:
                    break
                w = _Window(seq_len, lane, policy, isolate=False)
            offset += w.place_continuation(s, offset)
        used += 1
    if len(seqs) < max_seqs and w.cursor > 0:
        seqs.append(w.finalize())
    return PackResult(seqs[:max_seqs], used, policy)


def _pack_greedy(samples, seq_len, max_seqs, lane, policy, isolate=True):
    """First fit: each sample goes into the first open window with room.

    Order dependent by construction, which is the honest characterisation --
    the same multiset of samples in a different order gives different windows.
    """
    windows: List[_Window] = []
    used = 0
    for s in samples:
        take = min(s.length, seq_len)
        if isolate:
            take = _truncate_at_span_boundary(s, seq_len)
        placed = False
        for w in windows:
            if w.can_hold(take):
                w.place(s, limit=take)
                placed = True
                break
        if not placed:
            if len(windows) >= max_seqs:
                break
            w = _Window(seq_len, lane, policy, isolate=isolate)
            w.place(s, limit=take)
            windows.append(w)
        used += 1
    return PackResult([w.finalize() for w in windows[:max_seqs]], used, policy)


def _pack_best_fit(samples, seq_len, max_seqs, lane, policy, isolate=True):
    """Longest first, into the tightest window that still fits.

    Sorting by length descending and then choosing the tightest fit is the
    standard bin-packing heuristic and beats first-fit whenever the length
    distribution is wide -- which it is here, from 90-token forum posts to
    400-token reasoning traces.
    """
    order = sorted(range(len(samples)),
                   key=lambda i: (-samples[i].length, samples[i].doc_id))
    windows: List[_Window] = []
    used = 0
    for i in order:
        s = samples[i]
        take = _truncate_at_span_boundary(s, seq_len) if isolate else min(s.length, seq_len)
        candidates = [w for w in windows if w.can_hold(take)]
        if candidates:
            # tightest remaining space that still fits
            target = min(candidates, key=lambda w: w.room)
            target.place(s, limit=take)
        else:
            if len(windows) >= max_seqs:
                break
            w = _Window(seq_len, lane, policy, isolate=isolate)
            w.place(s, limit=take)
            windows.append(w)
        used += 1
    return PackResult([w.finalize() for w in windows[:max_seqs]], used, policy)


def _pack_structure_preserving(samples, seq_len, max_seqs, lane, policy):
    """Best-fit placement plus hard isolation between co-tenants.

    Same packing decisions as best-fit; different guarantees. Each sample is its
    own attention segment with its own position origin, and no gradient is ever
    taken across the join. That is what makes it safe to put two unrelated
    trajectories in one window.
    """
    return _pack_best_fit(samples, seq_len, max_seqs, lane, policy, isolate=True)


def _pack_long_context(samples, seq_len, max_seqs, lane, policy):
    """Long samples get a window to themselves; short ones are bucketed behind them.

    "Long-context batches are expensive. Every unused position wastes a
    high-value training opportunity." A long sample sharing a window with a
    short one wastes the expensive resource, so long samples are placed first
    and alone, and short samples fill whatever windows remain.
    """
    threshold = int(C.LONG_CONTEXT_FRACTION * seq_len)
    long_ones = [s for s in samples if s.length >= threshold]
    short_ones = [s for s in samples if s.length < threshold]

    windows: List[_Window] = []
    used = 0
    for s in long_ones:
        if len(windows) >= max_seqs:
            break
        w = _Window(seq_len, lane, policy, isolate=True)
        w.place(s, limit=_truncate_at_span_boundary(s, seq_len))
        windows.append(w)
        used += 1

    for s in short_ones:
        take = _truncate_at_span_boundary(s, seq_len)
        candidates = [w for w in windows if w.can_hold(take)]
        if candidates:
            min(candidates, key=lambda w: w.room).place(s, limit=take)
        else:
            if len(windows) >= max_seqs:
                break
            w = _Window(seq_len, lane, policy, isolate=True)
            w.place(s, limit=take)
            windows.append(w)
        used += 1

    return PackResult([w.finalize() for w in windows[:max_seqs]], used, policy)


_POLICIES = {
    "pad_only": _pack_pad_only,
    "concat_and_chop": _pack_concat_and_chop,
    "greedy": _pack_greedy,
    "best_fit": _pack_best_fit,
    "structure_preserving": _pack_structure_preserving,
    "long_context": _pack_long_context,
}

ISOLATING_POLICIES = {"pad_only", "greedy", "best_fit",
                      "structure_preserving", "long_context"}


def policy_for(lane: str, samples: Sequence[Sample], seq_len: int) -> str:
    """Lane's declared policy, with a long-context override.

    A lane normally packed by concatenation still needs the long-context policy
    when the sample in hand is long enough that a shared window would waste the
    expensive slots.
    """
    base = C.PACKING_POLICY.get(lane, "concat_and_chop")
    if samples:
        threshold = C.LONG_CONTEXT_FRACTION * seq_len
        longest = max(s.length for s in samples)
        if longest >= threshold and base in ("best_fit", "structure_preserving"):
            return "long_context"
    return base


def rebuild(seq_len: int, lane: str, policy: str, members: Sequence[dict],
            fetch) -> PackedSequence:
    """Reconstruct a window from recorded member coordinates alone.

    This is the replay primitive. `members` comes straight out of a ledger
    event and carries only coordinates -- shard id, offset within the shard,
    length, offset within the window, segment id. `fetch` re-reads the tokens
    and loss flags from the immutable shard.

    It deliberately does NOT re-run the packer. Re-running the packer would
    reconstruct a window from the same INPUTS, which proves the packer is
    deterministic but says nothing about whether the shard still holds the same
    bytes. Rebuilding from coordinates proves the stronger thing: the tokens
    that trained the model at that step are still there, unchanged, and can be
    produced again. It also works for concatenating policies, where a document
    is split across two windows and re-packing from whole documents would not
    reproduce either of them.
    """
    isolate = policy in ISOLATING_POLICIES
    w = _Window(seq_len, lane, policy, isolate)
    for idx, m in enumerate(members):
        tokens, loss = fetch(m["shard_id"], m["shard_start"], m["length"])
        if len(tokens) != m["length"]:
            raise ValueError(
                f"replay: shard {m['shard_id']} returned {len(tokens)} tokens "
                f"for a span of {m['length']} -- the shard has changed")
        off = m["offset"]
        for i in range(m["length"]):
            p = off + i
            w.ids[p] = tokens[i]
            w.loss[p] = loss[i]
            w.seg[p] = m["segment_id"]
            w.pos[p] = i if isolate else p
            w.docidx[p] = idx
        w.cursor = max(w.cursor, off + m["length"])
        # The ledger stores coordinates; the span id is derived from them, so a
        # replay reconstructs the same provenance string rather than trusting a
        # stored copy of it.
        w.members.append(dict(
            m, span_id=f"{m['shard_id']}:{m['shard_start']}:{m['length']}"))
        w.n_segments = max(w.n_segments, m["segment_id"])
    return w.finalize()


def attention_allows(seq: PackedSequence, i: int, j: int) -> bool:
    """Can position i attend to position j? Used by the model and the tests.

    Causal, never onto padding, and never across an isolation boundary.
    """
    if j > i:
        return False
    if seq.segment_ids[i] == 0 or seq.segment_ids[j] == 0:
        return False
    return seq.segment_ids[i] == seq.segment_ids[j]
