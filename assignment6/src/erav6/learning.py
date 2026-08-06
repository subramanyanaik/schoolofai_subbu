"""
learning.py — the second half of the two-way ledger.

"The consumption ledger records what the model saw. The learning ledger
attaches the outcome back to the data."

Two instruments live here.

THE TOKEN-LEVEL TRACE. For sampled steps, every loss-bearing token is recorded
with its decoded preview, position, document, shard, lane, and its own
cross-entropy and perplexity. This is the signal that cannot be recovered
later: reconstructing it would mean re-running the same model at the same
training state over the same data, which at real scale never happens cleanly.
Full traces are expensive, so the cadence is configurable and the run stores
aggregates for every step and full traces for a sample of them -- the tiered
strategy Session 6 describes.

THE PER-SHARD REPORT CARD. For every shard the model trains on, the loss delta
is MEASURED, not inferred: a fixed probe window is cut from the shard, the
model is evaluated on it immediately before the optimizer step and again
immediately after, and the difference is recorded against that shard. Same
probe, same window, same mask, two forward passes either side of one update.
That is what lets the run say "this shard helped" with a number behind it, and
what makes the useful/neutral/harmful classification something other than an
opinion.

The probe is deliberately a FIXED slice of the shard rather than the batch that
was just consumed. Measuring on the consumed batch would measure memorisation
of those exact tokens; measuring on a held-back slice of the same shard
measures whether anything transferred.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from . import config as C
from . import packing as P
from .catalog import Catalog
from .packing import PackedSequence, Sample

USEFUL, NEUTRAL, HARMFUL = "useful", "neutral", "harmful"
# Relative change in probe loss below which an exposure is called neutral.
NEUTRAL_BAND = 0.002


def model_phase(stage: str, step: int, total_steps: int) -> str:
    if stage.startswith("S4"):
        return "anneal"
    frac = step / max(1, total_steps - 1)
    if frac < 0.34:
        return "early"
    if frac < 0.67:
        return "mid"
    return "late"


class ProbeSet:
    """One fixed evaluation window per shard, cut once and reused all run.

    Reusing the same window every time is what makes deltas comparable across
    steps: a probe that moved would confound "the model improved" with "this
    slice was easier".
    """

    def __init__(self, cat: Catalog, seq_len: int, probe_tokens: int):
        self.cat = cat
        self.seq_len = seq_len
        self.probe_tokens = probe_tokens
        self._cache: Dict[str, Optional[PackedSequence]] = {}

    def probe(self, shard_id: str) -> Optional[PackedSequence]:
        if shard_id in self._cache:
            return self._cache[shard_id]
        seq = self._build(shard_id)
        self._cache[shard_id] = seq
        return seq

    def _build(self, shard_id: str) -> Optional[PackedSequence]:
        try:
            shard = self.cat.shard(shard_id)
        except (PermissionError, KeyError):
            return None
        if not shard.docs:
            return None
        # Last document of the shard: the draw order is shuffled per epoch, so
        # no document is systematically the probe, but the choice is fixed.
        doc = shard.docs[-1]
        toks = shard.tokens()
        start, length = doc["start"], min(doc["length"], self.probe_tokens)
        loss = [0] * doc["length"]
        for sp in doc["spans"]:
            if sp["loss"]:
                lo = sp["start"] - doc["start"]
                for i in range(lo, lo + sp["length"]):
                    if 0 <= i < doc["length"]:
                        loss[i] = 1
        sample = Sample(shard_id=shard_id, doc_id=doc["doc_id"], lane=shard.lane,
                        start=start, length=length,
                        tokens=toks[start:start + length],
                        loss=loss[:length], spans=doc["spans"])
        res = P.pack([sample], self.seq_len, "pad_only", max_seqs=1, lane=shard.lane)
        if not res.sequences or res.sequences[0].n_loss_tokens == 0:
            return None
        return res.sequences[0]

    @torch.no_grad()
    def evaluate(self, shard_ids: Sequence[str], model, device) -> Dict[str, float]:
        """Mean probe loss for each shard, in one batched forward pass."""
        ids = [s for s in shard_ids if self.probe(s) is not None]
        if not ids:
            return {}
        seqs = [self.probe(s) for s in ids]
        was_training = model.training
        model.eval()
        t = model.stack(seqs, device)
        _, per_token = model.loss(t["input_ids"], t["labels"], t["loss_mask"],
                                  t["position_ids"], t["segment_ids"])
        mask = t["loss_mask"].to(per_token.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        vals = (per_token.sum(dim=1) / denom).tolist()
        if was_training:
            model.train()
        return dict(zip(ids, vals))


@dataclass
class ShardRecord:
    shard_id: str
    lane: str
    exposures: int = 0
    first_step: int = -1
    last_step: int = -1
    total_loss_tokens: int = 0
    sum_token_loss: float = 0.0
    loss_deltas: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    opus_scores: List[float] = field(default_factory=list)
    phases: List[str] = field(default_factory=list)
    repeated_pass: int = 0

    @property
    def mean_token_loss(self) -> float:
        return self.sum_token_loss / max(1, self.total_loss_tokens)

    @property
    def total_delta(self) -> float:
        return sum(self.loss_deltas)

    def classify(self) -> str:
        """useful / neutral / harmful, from the measured probe deltas.

        Positive delta means probe loss went DOWN across the update, i.e. the
        exposure helped. The neutral band exists because a delta smaller than
        the noise of a single step is not evidence of anything.
        """
        if not self.loss_deltas:
            return NEUTRAL
        total = self.total_delta
        scale = max(1e-6, abs(self.mean_token_loss))
        rel = total / scale
        if rel > NEUTRAL_BAND:
            return USEFUL
        if rel < -NEUTRAL_BAND:
            return HARMFUL
        return NEUTRAL

    def as_dict(self) -> dict:
        return dict(
            shard_id=self.shard_id, lane=self.lane,
            exposures=self.exposures, repeated_pass=self.repeated_pass,
            first_step=self.first_step, last_step=self.last_step,
            loss_bearing_tokens=self.total_loss_tokens,
            mean_token_loss=round(self.mean_token_loss, 6),
            mean_token_perplexity=round(math.exp(min(20.0, self.mean_token_loss)), 4),
            probe_loss_delta_total=round(self.total_delta, 6),
            probe_loss_delta_mean=round(
                self.total_delta / max(1, len(self.loss_deltas)), 6),
            n_probe_measurements=len(self.loss_deltas),
            mean_grad_norm=round(
                sum(self.grad_norms) / max(1, len(self.grad_norms)), 6),
            mean_opus_score=round(
                sum(self.opus_scores) / max(1, len(self.opus_scores)), 6),
            phases=sorted(set(self.phases)),
            usefulness=self.classify(),
        )


class LearningLedger:
    def __init__(self, cat: Catalog, tokenizer, total_steps: int):
        self.cat = cat
        self.tokenizer = tokenizer
        self.total_steps = total_steps
        self.shards: Dict[str, ShardRecord] = {}
        self.lane_loss: Dict[str, List[float]] = {}

    def record_exposure(self, shard_id: str, lane: str, step: int, stage: str,
                        n_loss_tokens: int, sum_loss: float, grad_norm: float,
                        opus_score: float, repeated_pass: int) -> ShardRecord:
        rec = self.shards.get(shard_id)
        if rec is None:
            rec = ShardRecord(shard_id=shard_id, lane=lane, first_step=step)
            self.shards[shard_id] = rec
        rec.exposures += 1
        rec.last_step = step
        rec.total_loss_tokens += n_loss_tokens
        rec.sum_token_loss += sum_loss
        rec.grad_norms.append(grad_norm)
        rec.opus_scores.append(opus_score)
        rec.phases.append(model_phase(stage, step, self.total_steps))
        rec.repeated_pass = max(rec.repeated_pass, repeated_pass)
        return rec

    def record_delta(self, shard_id: str, before: float, after: float) -> float:
        delta = before - after            # positive = loss went down = helped
        rec = self.shards.get(shard_id)
        if rec is not None:
            rec.loss_deltas.append(delta)
        return delta

    def record_lane_loss(self, lane: str, mean_loss: float) -> None:
        self.lane_loss.setdefault(lane, []).append(mean_loss)

    # -- token-level trace -------------------------------------------------
    def token_trace(self, seq: PackedSequence, per_token_loss: Sequence[float],
                    step: int, stage: str, checkpoint_before: str,
                    checkpoint_after: str, opus_score: float,
                    repeated_pass: int, tokens_consumed: int,
                    max_records: int = 48) -> List[dict]:
        """Per-token records for the loss-bearing positions of one window."""
        out: List[dict] = []
        owner: Dict[int, dict] = {}
        for m in seq.members:
            for i in range(m["offset"], m["offset"] + m["length"]):
                owner[i] = m

        for i in range(seq.seq_len):
            if not seq.loss_mask[i]:
                continue
            if len(out) >= max_records:
                break
            target = seq.labels[i]
            loss = float(per_token_loss[i])
            m = owner.get(i, {})
            out.append(dict(
                step=step,
                position=i,
                token_id=int(target),
                token_preview=self.tokenizer.preview(int(target)),
                document_id=m.get("doc_id"),
                shard_id=m.get("shard_id"),
                lane=seq.lane,
                is_special=self.tokenizer.is_special(int(target)),
                is_boundary=bool(int(target) == C.EOS_ID),
                loss_mask=1,
                cross_entropy=round(loss, 6),
                perplexity=round(math.exp(min(20.0, loss)), 4),
                curriculum_stage=stage,
                model_phase=model_phase(stage, step, self.total_steps),
                tokens_consumed_when_seen=tokens_consumed,
                checkpoint_before=checkpoint_before,
                checkpoint_after=checkpoint_after,
                opus_score=round(opus_score, 6),
                repeated_pass=repeated_pass,
            ))
        return out

    # -- reporting ---------------------------------------------------------
    def report(self) -> dict:
        rows = [r.as_dict() for r in sorted(self.shards.values(),
                                            key=lambda x: x.shard_id)]
        by_class: Dict[str, int] = {}
        for r in rows:
            by_class[r["usefulness"]] = by_class.get(r["usefulness"], 0) + 1

        by_lane: Dict[str, dict] = {}
        for r in rows:
            lane = r["lane"]
            b = by_lane.setdefault(lane, dict(shards=0, loss_tokens=0,
                                              sum_delta=0.0, sum_loss=0.0))
            b["shards"] += 1
            b["loss_tokens"] += r["loss_bearing_tokens"]
            b["sum_delta"] += r["probe_loss_delta_total"]
            b["sum_loss"] += r["mean_token_loss"]
        for lane, b in by_lane.items():
            b["mean_token_loss"] = round(b["sum_loss"] / max(1, b["shards"]), 6)
            b["total_probe_delta"] = round(b["sum_delta"], 6)
            b.pop("sum_loss")
            b.pop("sum_delta")

        # The proxy-bias check Session 6 asks for: did OPUS systematically
        # under-score a lane that the measured deltas say was useful?
        bias: List[dict] = []
        for r in rows:
            if r["mean_opus_score"] < 0 and r["usefulness"] == USEFUL:
                bias.append(dict(shard_id=r["shard_id"], lane=r["lane"],
                                 mean_opus_score=r["mean_opus_score"],
                                 probe_delta=r["probe_loss_delta_total"]))

        return dict(
            n_shards_trained=len(rows),
            usefulness_counts=by_class,
            by_lane=by_lane,
            proxy_undervalued=bias,
            shards=rows,
        )
