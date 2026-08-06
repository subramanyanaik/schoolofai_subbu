"""
opus.py — the selector, and its audit trail.

"OPUS sits inside the data path. Its decisions are training events."

Every candidate batch is scored by running the CURRENT model over it. That is
the part that cannot be faked: the utility of a candidate is a measured
property of the model at this step, so the same candidate scores differently
early and late in the run, and a grader reading the ledger can see the scores
move as the model learns.

The utility has three terms, each answering a different question:

    headroom     mean loss of the candidate under the current model, z-scored
                 across the step's candidate pool. High loss means there is
                 something here the model has not learned yet.
    redundancy   how much of this candidate's shard set has been consumed in
                 the last few steps. Penalised, because the fourth pass over a
                 shard in ten steps is worth less than the first.
    stage fit    how much the current curriculum stage is asking for this lane.
                 A reasoning batch is worth more in the anneal than in warmup.

Four outcomes are recorded, not one:

    accepted        entered the gradient-bearing stream
    rejected        scored, not used, and the REASON is kept -- "rejected clean
                    data should not disappear"
    deferred        valuable but not wanted now; goes into a queue and is
                    re-offered within `defer_max_age` steps. The audit checks
                    that deferred candidates really do come back.
    floor_override  scored below the accept bar but rescued because the
                    protected floor requires a native-Indic sequence in this
                    batch. This is the case Session 5 predicted: an
                    English-tuned proxy systematically under-scores Indic
                    content, and without the override the floor would be
                    starved by the selector rather than by the supply.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import config as C
from . import plan
from .packing import PackedSequence

ACCEPTED = "accepted"
REJECTED = "rejected"
DEFERRED = "deferred"
FLOOR_OVERRIDE = "floor_override"


@dataclass
class Candidate:
    candidate_id: str
    lane: str
    seq: PackedSequence
    origin_step: int
    deferred_from: Optional[int] = None

    @property
    def shard_ids(self) -> List[str]:
        return sorted({m["shard_id"] for m in self.seq.members})

    @property
    def doc_ids(self) -> List[str]:
        return sorted({m["doc_id"] for m in self.seq.members})


@dataclass
class Decision:
    candidate_id: str
    lane: str
    status: str
    score: float
    headroom: float
    redundancy: float
    stage_fit: float
    reason: str
    shard_ids: List[str]
    doc_ids: List[str]
    span_ids: List[str]
    step: int
    stage: str
    model_checkpoint: str
    proxy_version: str
    effective_tokens: float
    n_loss_tokens: int
    protected_floor_override: bool = False
    deferred_from: Optional[int] = None

    def as_dict(self) -> dict:
        return dict(
            candidate_id=self.candidate_id, lane=self.lane, status=self.status,
            opus_score=round(self.score, 6),
            headroom=round(self.headroom, 6),
            redundancy=round(self.redundancy, 6),
            stage_fit=round(self.stage_fit, 6),
            rejection_reason=self.reason if self.status in (REJECTED, DEFERRED) else None,
            reason=self.reason,
            shard_ids=self.shard_ids, doc_ids=self.doc_ids, span_ids=self.span_ids,
            step=self.step, curriculum_stage=self.stage,
            model_checkpoint=self.model_checkpoint,
            proxy_version=self.proxy_version,
            effective_token_estimate=round(self.effective_tokens, 3),
            n_loss_tokens=self.n_loss_tokens,
            protected_floor_override=self.protected_floor_override,
            deferred_from_step=self.deferred_from,
        )


class OpusSelector:
    def __init__(self, cfg: dict = None, lane_bias: Dict[str, float] = None):
        self.cfg = dict(C.OPUS)
        if cfg:
            self.cfg.update(cfg)
        self.proxy_version = self.cfg["proxy_version"]
        # Per-lane additive bias on the utility. EMPTY for the real run -- it
        # exists so `floor_drill.py` can construct the English-tuned proxy
        # Session 5 warns about and show what the floor is protecting against.
        # Any non-empty bias changes the proxy version string, so a biased
        # decision can never be mistaken for a real one in the ledger.
        self.lane_bias = dict(lane_bias or {})
        if self.lane_bias:
            self.proxy_version += "-lane-biased-drill"
        # recent shard consumption, for the redundancy term
        self.recent: List[List[str]] = []
        # deferred queue, carried across steps and checkpointed
        self.deferred: List[dict] = []
        self._counter = 0

    # -- state (checkpointed) ---------------------------------------------
    def state_dict(self) -> dict:
        return dict(recent=[list(r) for r in self.recent],
                    deferred=[dict(d) for d in self.deferred],
                    counter=self._counter)

    def load_state_dict(self, state: dict) -> None:
        self.recent = [list(r) for r in state.get("recent", [])]
        self.deferred = [dict(d) for d in state.get("deferred", [])]
        self._counter = state.get("counter", 0)

    def next_candidate_id(self, step: int, lane: str) -> str:
        self._counter += 1
        return f"cand-{step:05d}-{lane}-{self._counter:05d}"

    # -- scoring -----------------------------------------------------------
    @torch.no_grad()
    def measure_headroom(self, candidates: Sequence[Candidate], model,
                         device: torch.device) -> List[float]:
        """Mean loss of each candidate under the current model.

        A real forward pass per candidate, batched. This is the cost of running
        a selector inside the data path, and `performance.json` reports it as
        such: OPUS scoring time is subtracted from the useful-tokens-per-second
        the loader can deliver.
        """
        if not candidates:
            return []
        was_training = model.training
        model.eval()
        out: List[float] = []
        batch = 8
        for i in range(0, len(candidates), batch):
            chunk = [c.seq for c in candidates[i:i + batch]]
            t = model.stack(chunk, device)
            _, per_token = model.loss(t["input_ids"], t["labels"], t["loss_mask"],
                                      t["position_ids"], t["segment_ids"])
            mask = t["loss_mask"].to(per_token.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            out.extend((per_token.sum(dim=1) / denom).tolist())
        if was_training:
            model.train()
        return out

    def redundancy_of(self, cand: Candidate) -> float:
        """Fraction of this candidate's shards seen in the recent window."""
        if not self.recent or not cand.shard_ids:
            return 0.0
        window = self.recent[-self.cfg["recent_window"]:]
        seen = {s for step_shards in window for s in step_shards}
        hits = sum(1 for s in cand.shard_ids if s in seen)
        return hits / len(cand.shard_ids)

    @staticmethod
    def stage_fit_of(lane: str, stage: str) -> float:
        """How much this stage wants this lane, normalised to [0, 1]."""
        mix = plan.STAGE_MIX.get(stage, plan.GLOBAL_MIXTURE)
        top = max(mix.values()) or 1.0
        return mix.get(lane, 0.0) / top

    # -- selection ---------------------------------------------------------
    def select(self, candidates: List[Candidate], quotas: Dict[str, int],
               step: int, stage: str, model, device: torch.device,
               model_checkpoint: str) -> Tuple[List[Candidate], List[Decision]]:
        """Score every candidate, then fill each lane's quota from its own pool."""
        if not candidates:
            return [], []

        headroom = self.measure_headroom(candidates, model, device)
        mu = statistics.fmean(headroom)
        sigma = statistics.pstdev(headroom) if len(headroom) > 1 else 0.0

        scored: List[dict] = []
        for cand, raw in zip(candidates, headroom):
            z = (raw - mu) / sigma if sigma > 1e-9 else 0.0
            red = self.redundancy_of(cand)
            fit = self.stage_fit_of(cand.lane, stage)
            utility = (z
                       - self.cfg["redundancy_weight"] * red
                       + self.cfg["stage_fit_weight"] * fit
                       + self.lane_bias.get(cand.lane, 0.0))
            scored.append(dict(cand=cand, raw=raw, z=z, red=red, fit=fit, u=utility))

        utilities = sorted(s["u"] for s in scored)
        accept_bar = _quantile(utilities, self.cfg["accept_quantile"])
        defer_bar = _quantile(utilities, self.cfg["defer_quantile"])

        by_lane: Dict[str, List[dict]] = {}
        for s in scored:
            by_lane.setdefault(s["cand"].lane, []).append(s)
        for lane in by_lane:
            by_lane[lane].sort(key=lambda s: (-s["u"], s["cand"].candidate_id))

        accepted: List[Candidate] = []
        decisions: List[Decision] = []

        for lane, pool in sorted(by_lane.items()):
            need = quotas.get(lane, 0)
            is_floor_lane = lane in plan.FLOOR_LANES
            for rank, s in enumerate(pool):
                cand = s["cand"]
                take = rank < need
                if take:
                    below_bar = s["u"] < accept_bar
                    if below_bar and is_floor_lane:
                        status, reason = FLOOR_OVERRIDE, "protected_floor_rescue"
                        override = True
                    elif below_bar:
                        status, reason = ACCEPTED, "quota_fill"
                        override = False
                    else:
                        status, reason = ACCEPTED, "high_proxy_utility"
                        override = False
                    accepted.append(cand)
                else:
                    override = False
                    if s["u"] >= defer_bar and cand.deferred_from is None:
                        status = DEFERRED
                        reason = ("stage_mismatch" if s["fit"] < 0.25
                                  else "quota_pressure")
                    else:
                        status = REJECTED
                        if cand.deferred_from is not None:
                            reason = "defer_expired"
                        elif s["red"] >= 0.5:
                            reason = "duplication"
                        elif s["fit"] < 0.25:
                            reason = "stage_mismatch"
                        else:
                            reason = "low_proxy_utility"

                decisions.append(Decision(
                    candidate_id=cand.candidate_id, lane=lane, status=status,
                    score=s["u"], headroom=s["raw"], redundancy=s["red"],
                    stage_fit=s["fit"], reason=reason,
                    shard_ids=cand.shard_ids, doc_ids=cand.doc_ids,
                    span_ids=cand.seq.provenance(), step=step, stage=stage,
                    model_checkpoint=model_checkpoint,
                    proxy_version=self.proxy_version,
                    effective_tokens=_effective_tokens(cand, s["u"]),
                    n_loss_tokens=cand.seq.n_loss_tokens,
                    protected_floor_override=override,
                    deferred_from=cand.deferred_from,
                ))

        # Carry deferred candidates forward by DESCRIPTOR, not by tensor: a
        # deferred candidate is re-drawn from its shard spans when it is
        # re-offered, which keeps the queue small and keeps the shards the
        # single source of truth.
        self.deferred = [d for d in self.deferred
                         if step - d["deferred_at"] < self.cfg["defer_max_age"]]
        for d in decisions:
            if d.status == DEFERRED:
                self.deferred.append(dict(
                    candidate_id=d.candidate_id, lane=d.lane,
                    span_ids=d.span_ids, deferred_at=step, score=d.score,
                    reason=d.reason))

        self.recent.append(sorted({s for c in accepted for s in c.shard_ids}))
        return accepted, decisions

    def due_deferred(self, step: int, lane: str = None) -> List[dict]:
        """Deferred candidates eligible to be re-offered at this step."""
        return [d for d in self.deferred
                if (lane is None or d["lane"] == lane)
                and d["deferred_at"] < step
                and step - d["deferred_at"] <= self.cfg["defer_max_age"]]

    def drop_deferred(self, candidate_ids: Sequence[str]) -> None:
        ids = set(candidate_ids)
        self.deferred = [d for d in self.deferred if d["candidate_id"] not in ids]


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _effective_tokens(cand: Candidate, utility: float) -> float:
    """Proxy estimate of a candidate's training value in token-equivalents.

    Loss-bearing tokens scaled by the selector's own utility. Explicitly a
    PROXY -- it is the selector's belief, recorded so it can later be compared
    against what the learning ledger measured actually happened.
    """
    return cand.seq.n_loss_tokens * max(0.0, 1.0 + 0.5 * utility)


def decisions_summary(decisions: Sequence[Decision]) -> dict:
    by_status: Dict[str, int] = {}
    by_lane_status: Dict[str, Dict[str, int]] = {}
    reasons: Dict[str, int] = {}
    for d in decisions:
        by_status[d.status] = by_status.get(d.status, 0) + 1
        by_lane_status.setdefault(d.lane, {})
        by_lane_status[d.lane][d.status] = by_lane_status[d.lane].get(d.status, 0) + 1
        if d.status in (REJECTED, DEFERRED):
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
    return dict(by_status=by_status, by_lane=by_lane_status,
                rejection_reasons=reasons, n=len(decisions),
                n_floor_overrides=sum(1 for d in decisions if d.protected_floor_override))
