"""
dataloader.py — the planned stream, and the state needed to resume it.

The contract this module has to meet is narrow and absolute:

    the batch at (branch, seed, step) is a pure function of the plan, the
    admitted shard pool, and the loader state carried in the checkpoint.

Everything about crash recovery follows from that. If the batch depends on
anything else -- wall-clock time, dict iteration order, how many worker
processes happened to be alive -- then a resumed run silently sees a different
data stream, and "resume worked" becomes unverifiable. Session 6 puts it as
`experiment = model checkpoint + optimizer state + data stream + code/config`.

The loader state that must survive a crash is larger than it first looks:

    next_step      obvious
    lane_cursor    where each lane is in its shuffled sample order
    lane_epoch     how many passes over the lane have completed; also selects
                   the shuffle, so getting it wrong changes the ORDER, not just
                   the position
    carry          samples drawn from a shard but not yet packed into an
                   emitted window. Dropping these on resume would skip data;
                   re-drawing them would repeat it.
    opus state     the deferred queue and the recent-shard window

Forgetting `carry` is the classic resume bug: everything looks right at the
step boundary and a handful of samples quietly vanish. It is checkpointed here
as descriptors, and the resume proof compares full batch content hashes rather
than batch ids, so a lost carry buffer would be caught.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config as C
from . import packing as P
from . import plan
from .catalog import Catalog
from .hashing import hash_obj, sha256_text
from .mixture import Timeline
from .opus import Candidate, Decision, OpusSelector
from .packing import PackedSequence, Sample


# ---------------------------------------------------------------------------
# Batch produced for one optimizer step
# ---------------------------------------------------------------------------
@dataclass
class Microbatch:
    rank: int
    microbatch_id: int
    batch_id: str
    sequences: List[PackedSequence]

    def content_hash(self) -> str:
        return sha256_text("\n".join(s.content_hash() for s in self.sequences))

    def loss_mask_hash(self) -> str:
        return sha256_text("\n".join(s.loss_mask_hash() for s in self.sequences))

    def span_ids(self) -> List[str]:
        return [sid for s in self.sequences for sid in s.provenance()]

    def shard_ids(self) -> List[str]:
        return sorted({m["shard_id"] for s in self.sequences for m in s.members})

    def doc_ids(self) -> List[str]:
        return sorted({m["doc_id"] for s in self.sequences for m in s.members})


@dataclass
class StepBatch:
    step: int
    stage: str
    branch: str
    microbatches: List[Microbatch]
    decisions: List[Decision] = field(default_factory=list)
    lane_counts: Dict[str, int] = field(default_factory=dict)
    substitutions: List[dict] = field(default_factory=list)
    scarcity: List[dict] = field(default_factory=list)
    floor_delivered: int = 0
    floor_required: int = 0

    @property
    def sequences(self) -> List[PackedSequence]:
        return [s for mb in self.microbatches for s in mb.sequences]

    def batch_id(self) -> str:
        return f"{self.branch}/{self.step:06d}"

    def content_hash(self) -> str:
        """Identity of the whole optimizer step's data.

        This is the value the resume proof compares. It covers every token,
        mask bit, segment id and position id in every microbatch, in order.
        """
        return sha256_text("\n".join(mb.content_hash() for mb in self.microbatches))

    def n_real_tokens(self) -> int:
        return sum(s.n_real_tokens for s in self.sequences)

    def n_loss_tokens(self) -> int:
        return sum(s.n_loss_tokens for s in self.sequences)

    def n_positions(self) -> int:
        return sum(s.seq_len for s in self.sequences)

    def utilization(self) -> float:
        pos = self.n_positions()
        return self.n_real_tokens() / pos if pos else 0.0


# ---------------------------------------------------------------------------
# Loader state
# ---------------------------------------------------------------------------
@dataclass
class LoaderState:
    branch: str
    data_seed: int
    next_step: int = 0
    lane_cursor: Dict[str, int] = field(default_factory=dict)
    lane_epoch: Dict[str, int] = field(default_factory=dict)
    carry: Dict[str, List[List[str]]] = field(default_factory=dict)
    scarcity: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(branch=self.branch, data_seed=self.data_seed,
                    next_step=self.next_step,
                    lane_cursor=dict(self.lane_cursor),
                    lane_epoch=dict(self.lane_epoch),
                    carry={k: [list(x) for x in v] for k, v in self.carry.items()},
                    scarcity=list(self.scarcity))

    @staticmethod
    def from_dict(d: dict) -> "LoaderState":
        return LoaderState(
            branch=d["branch"], data_seed=d["data_seed"],
            next_step=d["next_step"],
            lane_cursor=dict(d.get("lane_cursor", {})),
            lane_epoch=dict(d.get("lane_epoch", {})),
            carry={k: [list(x) for x in v] for k, v in d.get("carry", {}).items()},
            scarcity=list(d.get("scarcity", [])),
        )

    def fingerprint(self) -> str:
        return hash_obj(self.as_dict())


def _order_seed(data_seed: int, branch: str, lane: str, epoch: int) -> int:
    """Deterministic shuffle seed. Branch is in the key, so a fork really does
    get a different stream while remaining reproducible on its own terms."""
    return int(sha256_text(f"{data_seed}|{branch}|{lane}|{epoch}")[:15], 16)


class DataLoader:
    def __init__(self, cat: Catalog, timeline: Timeline, profile: dict,
                 branch: str, data_seed: int, selector: OpusSelector = None,
                 state: LoaderState = None, meter=None):
        self.meter = meter
        self.cat = cat
        self.timeline = timeline
        self.profile = profile
        self.seq_len = profile["seq_len"]
        self.microbatch = profile["microbatch"]
        self.n_ranks = profile["n_ranks"]
        self.grad_accum = profile["grad_accum"]
        self.seqs_per_step = self.n_ranks * self.microbatch * self.grad_accum
        self.selector = selector or OpusSelector()
        self.state = state or LoaderState(branch=branch, data_seed=data_seed)
        self._order_cache: Dict[Tuple[str, int], List[List[str]]] = {}
        self._reserved = {s for ids in timeline.reserved_shards.values() for s in ids}

        # Per-lane read counters, for the throughput report.
        self.shard_reads = 0
        self.cache_hits = 0

    # -- sample ordering ---------------------------------------------------
    def lane_order(self, lane: str, epoch: int) -> List[List[str]]:
        key = (lane, epoch)
        if key in self._order_cache:
            return self._order_cache[key]
        items: List[List[str]] = []
        for sid in sorted(self.cat.by_lane.get(lane, [])):
            for d in self.cat.shards[sid].docs:
                items.append([sid, d["doc_id"]])
        rng = random.Random(_order_seed(self.state.data_seed, self.state.branch, lane, epoch))
        rng.shuffle(items)
        self._order_cache[key] = items
        return items

    def _materialize(self, shard_id: str, doc_id: str) -> Sample:
        """Cut one document out of its immutable shard."""
        shard = self.cat.shard(shard_id)
        cached = shard._cache is not None
        toks = shard.tokens()
        self.shard_reads += 1
        if cached:
            self.cache_hits += 1
        d = shard.doc(doc_id)
        start, length = d["start"], d["length"]
        loss = [0] * length
        for sp in d["spans"]:
            if sp["loss"]:
                lo = sp["start"] - start
                for i in range(lo, lo + sp["length"]):
                    if 0 <= i < length:
                        loss[i] = 1
        return Sample(shard_id=shard_id, doc_id=doc_id, lane=shard.lane,
                      start=start, length=length,
                      tokens=toks[start:start + length], loss=loss,
                      spans=d["spans"])

    def _draw(self, lane: str, reserve_unlocked: bool) -> Optional[Sample]:
        """Next sample for `lane`, honouring carry, reserve and epoch caps."""
        carry = self.state.carry.setdefault(lane, [])
        if carry:
            sid, did = carry.pop(0)
            return self._materialize(sid, did)

        order = self.lane_order(lane, self.state.lane_epoch.get(lane, 0))
        if not order:
            return None

        guard = 0
        while guard <= len(order) * 2 + 4:
            guard += 1
            cur = self.state.lane_cursor.get(lane, 0)
            if cur >= len(order):
                if not self._roll_epoch(lane):
                    return None
                order = self.lane_order(lane, self.state.lane_epoch.get(lane, 0))
                continue
            sid, did = order[cur]
            self.state.lane_cursor[lane] = cur + 1
            # Anneal reserve: scarce Tier-A shards are invisible until S4.
            if sid in self._reserved and not reserve_unlocked:
                continue
            return self._materialize(sid, did)
        return None

    def _roll_epoch(self, lane: str) -> bool:
        """Start another pass over a lane, applying its declared fallback.

        Returns False when the lane must not be read again this run.
        """
        nxt = self.state.lane_epoch.get(lane, 0) + 1
        cap = plan.MAX_EPOCHS.get(lane, 1.0) * C.EPOCH_CAP_SCALE
        fallback = C.SCARCITY_FALLBACK.get(lane, "repeat")
        over_cap = nxt >= cap

        event = dict(lane=lane, epoch=nxt, cap=round(cap, 4),
                     step=self.state.next_step, fallback=fallback,
                     over_cap=bool(over_cap))

        if not over_cap or fallback == "repeat":
            event["action"] = "repeat"
            event["applied"] = True
            self.state.lane_epoch[lane] = nxt
            self.state.lane_cursor[lane] = 0
            self.state.scarcity.append(event)
            return True

        if fallback == "synthesize":
            # Generation is only defensible where an independent verifier
            # exists (Session 5 section 2.2). The derived synthetic supply that
            # HAS such a gate was materialised at catalog build time and is
            # already in the pool; there is nothing further to generate inside
            # the run, so the honest action is to stop drawing this lane and
            # say so, rather than to quietly repeat past the cap.
            event["action"] = "halt_lane_no_verifier_at_runtime"
            event["applied"] = True
            self.state.scarcity.append(event)
            return False

        # reduce_share
        event["action"] = "reduce_share"
        event["applied"] = True
        self.state.scarcity.append(event)
        return False

    # -- candidate assembly ------------------------------------------------
    def _candidates_for_lane(self, lane: str, n_seqs: int,
                             reserve_unlocked: bool, step: int) -> Tuple[List[Candidate], List[Sample]]:
        """Draw and pack enough sequences to offer OPUS a real choice."""
        if n_seqs <= 0:
            return [], []
        want = max(n_seqs, int(math.ceil(n_seqs * self.profile["opus_candidate_factor"])))
        drawn: List[Sample] = []
        sequences: List[PackedSequence] = []
        policy = C.PACKING_POLICY.get(lane, "concat_and_chop")

        while True:
            policy = P.policy_for(lane, drawn, self.seq_len)
            result = P.pack(drawn, self.seq_len, policy, max_seqs=want, lane=lane)
            sequences = result.sequences
            if len(sequences) >= want:
                used = result.n_samples_used
                break
            s = self._draw(lane, reserve_unlocked)
            if s is None:
                used = len(drawn)
                break
            drawn.append(s)

        sequences = sequences[:want]
        # Samples that ended up in no emitted window go back to carry, in order.
        placed_docs = {(m["shard_id"], m["doc_id"]) for q in sequences for m in q.members}
        leftover = [[s.shard_id, s.doc_id] for s in drawn
                    if (s.shard_id, s.doc_id) not in placed_docs]
        if leftover:
            self.state.carry.setdefault(lane, [])[0:0] = leftover

        cands = [Candidate(candidate_id=self.selector.next_candidate_id(step, lane),
                           lane=lane, seq=q, origin_step=step)
                 for q in sequences]
        return cands, drawn

    def _redraw_deferred(self, step: int, quotas: Dict[str, int]) -> List[Candidate]:
        """Re-offer deferred candidates by rebuilding them from their spans.

        The queue stores span coordinates, not tensors. Rebuilding from the
        immutable shards is the same operation replay uses, so a deferred
        candidate that comes back is byte-identical to the one that was
        deferred.
        """
        out: List[Candidate] = []
        for d in self.selector.due_deferred(step):
            if quotas.get(d["lane"], 0) <= 0:
                continue
            samples = []
            ok = True
            for span_id in d["span_ids"]:
                shard_id, start, length = _parse_span(span_id)
                try:
                    shard = self.cat.shard(shard_id)
                except PermissionError:
                    ok = False
                    break
                doc = next((x for x in shard.docs
                            if x["start"] <= start < x["start"] + x["length"]), None)
                if doc is None:
                    ok = False
                    break
                samples.append(self._materialize(shard_id, doc["doc_id"]))
            if not ok or not samples:
                continue
            policy = P.policy_for(d["lane"], samples, self.seq_len)
            res = P.pack(samples, self.seq_len, policy, max_seqs=1, lane=d["lane"])
            if not res.sequences:
                continue
            out.append(Candidate(candidate_id=d["candidate_id"], lane=d["lane"],
                                 seq=res.sequences[0], origin_step=step,
                                 deferred_from=d["deferred_at"]))
        self.selector.drop_deferred([c.candidate_id for c in out])
        return out

    # -- the step ----------------------------------------------------------
    def build_step(self, step: int, model, device, model_checkpoint: str) -> StepBatch:
        q = self.timeline.at(step)
        quotas = dict(q.seqs)
        reserve_unlocked = q.reserve_unlocked

        candidates: List[Candidate] = []
        substitutions: List[dict] = []
        shortfall_before = dict(quotas)

        pool_by_lane: Dict[str, List[Candidate]] = {}
        for lane in sorted(quotas):
            need = quotas[lane]
            if need <= 0:
                continue
            cands, _ = self._candidates_for_lane(lane, need, reserve_unlocked, step)
            pool_by_lane[lane] = cands
            candidates.extend(cands)

            served = min(len(cands), need)
            if served < need:
                # The lane could not fill its quota. Move the SHORTFALL -- not
                # the candidate count -- to whichever available lane has the
                # most unused supply. Adding the candidate count here inflates
                # the batch beyond `seqs_per_step`, and the surplus then gets
                # trimmed in lane-alphabetical order, which silently starves
                # whichever lanes sort last. That is how the protected floor
                # was being violated before this was fixed.
                missing = need - served
                donor = self._pick_substitute(lane, quotas, reserve_unlocked)
                if donor:
                    extra, _ = self._candidates_for_lane(donor, missing,
                                                         reserve_unlocked, step)
                    add = min(missing, len(extra))
                    if add:
                        quotas[donor] = quotas.get(donor, 0) + add
                        pool_by_lane.setdefault(donor, []).extend(extra)
                        candidates.extend(extra)
                        substitutions.append(dict(
                            step=step, from_lane=lane, to_lane=donor, seqs=add,
                            reason=C.SCARCITY_FALLBACK.get(lane, "repeat")))
            quotas[lane] = served

        candidates.extend(self._redraw_deferred(step, quotas))
        assert sum(quotas.values()) <= self.seqs_per_step, (
            "quota redistribution must never grow the global batch")

        # Timed separately: running a selector inside the data path costs
        # forward passes over candidates that are then discarded, and the
        # throughput report has to show that cost rather than hide it inside
        # "dataload".
        if self.meter is not None:
            with self.meter.timer("opus_scoring"):
                accepted, decisions = self.selector.select(
                    candidates, quotas, step, q.stage, model, device, model_checkpoint)
        else:
            accepted, decisions = self.selector.select(
                candidates, quotas, step, q.stage, model, device, model_checkpoint)
        accepted, decisions = self._enforce_floor(
            accepted, decisions, candidates, q.floor_required, step)

        # Deterministic order: lane, then candidate id. Never dict order.
        accepted.sort(key=lambda c: (c.lane, c.candidate_id))
        seqs = [c.seq for c in accepted][:self.seqs_per_step]

        lane_counts: Dict[str, int] = {}
        for s in seqs:
            lane_counts[s.lane] = lane_counts.get(s.lane, 0) + 1

        microbatches = self._assign(seqs, step)
        floor_delivered = sum(lane_counts.get(l, 0) for l in plan.FLOOR_LANES)

        self.state.next_step = step + 1
        return StepBatch(
            step=step, stage=q.stage, branch=self.state.branch,
            microbatches=microbatches, decisions=decisions,
            lane_counts=lane_counts, substitutions=substitutions,
            scarcity=[e for e in self.state.scarcity if e.get("step") == step],
            floor_delivered=floor_delivered, floor_required=q.floor_required)

    def _enforce_floor(self, accepted: List[Candidate], decisions: List[Decision],
                       candidates: List[Candidate], required: int,
                       step: int):
        """Rescue native-Indic sequences the selector declined to accept.

        The quota compiler reserves floor slots, but the selector can still end
        up delivering fewer -- a floor lane whose candidates all score badly,
        or whose quota was moved away by a substitution. Enforcing the floor
        only at quota time therefore leaves batches that miss it.

        This is the case Session 5 predicted rather than an implementation
        accident: "English-tuned quality proxies systematically reject Indic
        script, so an unprotected selector starves the lane." The fix is the
        override the assignment names -- take the best-scoring floor-lane
        candidate the selector passed over, drop the weakest accepted
        non-floor sequence to make room, and record BOTH decisions so the
        rescue and its cost are visible in the audit trail.
        """
        floor_lanes = set(plan.FLOOR_LANES)
        by_id = {d.candidate_id: d for d in decisions}
        accepted_ids = {c.candidate_id for c in accepted}

        while sum(1 for c in accepted if c.lane in floor_lanes) < required:
            spare = [c for c in candidates
                     if c.lane in floor_lanes and c.candidate_id not in accepted_ids]
            if not spare:
                break                      # the lane genuinely has nothing left
            rescue = max(spare, key=lambda c: (by_id[c.candidate_id].score
                                               if c.candidate_id in by_id else 0.0,
                                               c.candidate_id))
            displaceable = [c for c in accepted if c.lane not in floor_lanes]
            if not displaceable:
                break
            victim = min(displaceable,
                         key=lambda c: (by_id[c.candidate_id].score
                                        if c.candidate_id in by_id else 0.0,
                                        c.candidate_id))

            accepted.remove(victim)
            accepted_ids.discard(victim.candidate_id)
            accepted.append(rescue)
            accepted_ids.add(rescue.candidate_id)

            if rescue.candidate_id in by_id:
                d = by_id[rescue.candidate_id]
                d.status = OP.FLOOR_OVERRIDE
                d.reason = "protected_floor_rescue"
                d.protected_floor_override = True
            if victim.candidate_id in by_id:
                d = by_id[victim.candidate_id]
                d.status = OP.REJECTED
                d.reason = "displaced_by_protected_floor"

        return accepted, decisions

    def _pick_substitute(self, lane: str, quotas: Dict[str, int],
                         reserve_unlocked: bool) -> Optional[str]:
        """Lane with the most unused supply, excluding the starved one."""
        best, best_head = None, -1.0
        for cand in plan.LANES:
            if cand == lane or not self.cat.by_lane.get(cand):
                continue
            epoch = self.state.lane_epoch.get(cand, 0)
            cap = plan.MAX_EPOCHS.get(cand, 1.0) * C.EPOCH_CAP_SCALE
            headroom = cap - epoch
            if headroom > best_head:
                best, best_head = cand, headroom
        return best if best_head > 0 else None

    def _assign(self, seqs: List[PackedSequence], step: int) -> List[Microbatch]:
        """Deal sequences across ranks and gradient-accumulation slots.

        slot = i // microbatch, rank = slot % n_ranks, accum id = slot // n_ranks.
        Standard interleaving, and fixed, because the ledger records (rank,
        microbatch_id) and a replay has to reproduce the same partition.
        """
        out: List[Microbatch] = []
        n_slots = self.n_ranks * self.grad_accum
        buckets: List[List[PackedSequence]] = [[] for _ in range(n_slots)]
        for i, s in enumerate(seqs):
            buckets[min(i // self.microbatch, n_slots - 1)].append(s)
        for slot, group in enumerate(buckets):
            if not group:
                continue
            rank = slot % self.n_ranks
            mb_id = slot // self.n_ranks
            out.append(Microbatch(
                rank=rank, microbatch_id=mb_id,
                batch_id=f"{self.state.branch}/{step:06d}/r{rank}/m{mb_id}",
                sequences=group))
        return out

    # -- state -------------------------------------------------------------
    def state_dict(self) -> dict:
        return dict(loader=self.state.as_dict(), opus=self.selector.state_dict())

    def load_state_dict(self, d: dict) -> None:
        self.state = LoaderState.from_dict(d["loader"])
        self.selector.load_state_dict(d["opus"])
        self._order_cache.clear()


def _parse_span(span_id: str) -> Tuple[str, int, int]:
    shard_id, start, length = span_id.rsplit(":", 2)
    return shard_id, int(start), int(length)


def shard_span_reader(cat: Catalog):
    """Read (tokens, loss_flags) for an absolute span of an admitted shard.

    Loss flags are derived from the shard's own span index, not from anything
    remembered by the run. So a replay reconstructs the mask the same way the
    original did -- from the sealed object -- rather than trusting a stored copy.
    """
    def fetch(shard_id: str, start: int, length: int):
        shard = cat.shard(shard_id)
        tokens = shard.read_span(start, length)
        loss = [0] * length
        for d in shard.docs:
            if d["start"] + d["length"] <= start or d["start"] >= start + length:
                continue
            for sp in d["spans"]:
                if not sp["loss"]:
                    continue
                lo = max(sp["start"], start)
                hi = min(sp["start"] + sp["length"], start + length)
                for i in range(lo, hi):
                    loss[i - start] = 1
        return tokens, loss
    return fetch


def rebuild_sequences_from_ledger(cat: Catalog, event: dict,
                                  seq_len: int) -> List[PackedSequence]:
    """Reconstruct a consumed microbatch from its ledger event alone.

    Takes nothing from memory. The event names, for every window, the shard,
    the offset within it, the length, the offset within the window and the
    segment id; the tokens come back off disk from the immutable shard. If a
    shard changed, the rebuilt content hash will not match the recorded one --
    which is exactly what the replay check is for.
    """
    fetch = shard_span_reader(cat)
    return [
        P.rebuild(seq_len, rec["lane"], rec["policy"], rec["members"], fetch)
        for rec in event["sequences"]
    ]
