"""
mixture.py — the Session-5 curriculum compiled into per-step quotas.

Session 5 says "22% code". A dataloader cannot consume a percentage. This
module turns shares into an integer number of sequences per lane per optimizer
step, which is the only form the batch builder can actually execute.

Three things happen here that a naive `int(share * n)` would get wrong:

  1. LARGEST REMAINDER, not rounding. Rounding eleven lane shares independently
     does not produce a batch of the right size -- it produces 7 or 9 sequences
     when 8 were asked for. Largest remainder allocates the shortfall to the
     lanes with the biggest fractional parts, so the parts always sum to exactly
     the batch size and the cumulative drift against the plan stays bounded.

  2. THE PROTECTED FLOOR IS APPLIED AFTER ALLOCATION, BY PROMOTION. 8% of every
     batch must be native Indic. If the proportional allocation already delivers
     that, nothing happens. If it does not, sequences are taken from the lane
     furthest ABOVE its own target and given to the native-Indic lane furthest
     BELOW its own -- so the floor is paid for by whichever lane is currently
     over-served, not by a fixed victim. Each promotion is recorded.

  3. FEASIBILITY IS CHECKED AGAINST THE ADMITTED POOL, NOT THE CORPUS. A lane
     whose only shard was blocked by the firewall has zero available tokens, and
     the compiler must notice that at compile time rather than discovering it
     mid-run. Lanes that cannot be satisfied are reported with the fallback
     that Session 5 declared for them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from . import config as C
from . import plan
from .catalog import Catalog


def largest_remainder(shares: Dict[str, float], total: int) -> Dict[str, int]:
    """Allocate `total` integer slots across float shares with no drift.

    Ties are broken by lane name so the allocation is deterministic -- a batch
    that depends on dict iteration order would not replay.
    """
    if total <= 0:
        return {k: 0 for k in shares}
    pos = {k: max(0.0, v) for k, v in shares.items()}
    s = sum(pos.values())
    if s <= 0:
        return {k: 0 for k in shares}
    norm = {k: v / s for k, v in pos.items()}

    exact = {k: v * total for k, v in norm.items()}
    out = {k: int(exact[k]) for k in exact}
    short = total - sum(out.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))
    for k in order[:short]:
        out[k] += 1
    return out


class DeficitAllocator:
    """Cumulative-deficit allocation across steps.

    Per-step largest-remainder is the obvious approach and it is wrong at small
    batch sizes. With 8 sequences per step, a lane whose target share is 4% asks
    for 0.32 sequences; it loses the remainder contest to bigger lanes every
    single step and is allocated ZERO for the entire run. Measured on this
    timeline before the fix: indic_synthetic 0.000 realised against a 0.040
    target, forums and multilingual at a third of target.

    Carrying the fractional debt fixes it. Each step adds `share * n` to a
    lane's cumulative target, and sequences are handed to whichever lanes are
    furthest behind their own cumulative target. A 4%-share lane accumulates
    0.32 per step and is served on roughly every third step, which is the
    correct behaviour: it is delivered as a share of the RUN, not forced into
    every batch. Drift against the plan stays bounded by one sequence per lane
    instead of growing without limit.

    This is deterministic and history-dependent, which is fine: the timeline is
    compiled once from the plan, so a resumed run recompiles exactly the same
    quotas.
    """

    def __init__(self, lanes: List[str]):
        self.lanes = list(lanes)
        self.cum_target: Dict[str, float] = {l: 0.0 for l in lanes}
        self.cum_delivered: Dict[str, int] = {l: 0 for l in lanes}

    def deficit(self, lane: str) -> float:
        return self.cum_target[lane] - self.cum_delivered[lane]

    def allocate(self, eff_mix: Dict[str, float], n_seqs: int) -> Dict[str, int]:
        for lane in self.lanes:
            self.cum_target[lane] += eff_mix.get(lane, 0.0) * n_seqs

        seqs = {lane: 0 for lane in self.lanes}
        provisional = dict(self.cum_delivered)
        for _ in range(n_seqs):
            # Only lanes that are actually schedulable this step compete.
            eligible = [l for l in self.lanes if eff_mix.get(l, 0.0) > 0.0]
            if not eligible:
                break
            lane = max(eligible,
                       key=lambda l: (self.cum_target[l] - provisional[l], l))
            seqs[lane] += 1
            provisional[lane] += 1
        return seqs

    def commit(self, seqs: Dict[str, int]) -> None:
        """Record what was actually delivered, AFTER floor promotion."""
        for lane, n in seqs.items():
            self.cum_delivered[lane] = self.cum_delivered.get(lane, 0) + n


@dataclass
class StepQuota:
    step: int
    stage: str
    n_seqs: int
    target_mix: Dict[str, float]          # ramped shares actually asked for
    seqs: Dict[str, int]                  # integer sequences per lane
    floor_required: int
    floor_delivered: int
    floor_promotions: List[dict] = field(default_factory=list)
    unavailable_lanes: List[str] = field(default_factory=list)
    reserve_unlocked: bool = False

    def as_dict(self) -> dict:
        return dict(step=self.step, stage=self.stage, n_seqs=self.n_seqs,
                    target_mix={k: round(v, 6) for k, v in self.target_mix.items()},
                    seqs=dict(self.seqs),
                    floor_required=self.floor_required,
                    floor_delivered=self.floor_delivered,
                    floor_promotions=self.floor_promotions,
                    unavailable_lanes=self.unavailable_lanes,
                    reserve_unlocked=self.reserve_unlocked)


@dataclass
class Timeline:
    total_steps: int
    seqs_per_step: int
    seq_len: int
    stage_alloc: List[dict]
    quotas: List[StepQuota]
    feasibility: List[dict]
    reserved_shards: Dict[str, List[str]]

    def at(self, step: int) -> StepQuota:
        return self.quotas[step]

    def planned_lane_shares(self) -> Dict[str, float]:
        total = sum(q.n_seqs for q in self.quotas)
        acc: Dict[str, int] = {}
        for q in self.quotas:
            for lane, n in q.seqs.items():
                acc[lane] = acc.get(lane, 0) + n
        return {lane: acc.get(lane, 0) / total for lane in plan.LANES}

    def as_dict(self) -> dict:
        return dict(
            total_steps=self.total_steps,
            seqs_per_step=self.seqs_per_step,
            seq_len=self.seq_len,
            floor_pct=plan.FLOOR_PCT,
            floor_lanes=plan.FLOOR_LANES,
            stage_alloc=self.stage_alloc,
            ramp_steps=plan.RAMP_STEPS,
            session5_transitions=[
                dict(frm=t["frm"], to=t["to"], worst_lane=t["worst_lane"],
                     worst_delta=round(t["worst_delta"], 6),
                     ramp_tokens=t["ramp_tokens"], pp_per_100B=round(t["pp_per_100B"], 4),
                     within_limit=t["within_limit"])
                for t in plan.transition_report()
            ],
            planned_global_shares={k: round(v, 6) for k, v in self.planned_lane_shares().items()},
            session5_target_shares=plan.GLOBAL_MIXTURE,
            feasibility=self.feasibility,
            reserved_shards=self.reserved_shards,
            quotas=[q.as_dict() for q in self.quotas],
        )


def choose_reserved_shards(cat: Catalog) -> Dict[str, List[str]]:
    """Hold back the last shard of each anneal-reserved lane.

    Scarce Tier-A data that is drawn uniformly gets exhausted in the foundation
    stage, which is exactly when it is least valuable. Reserving it costs
    nothing before the anneal and is the difference between an anneal that has
    Tier-A material and one that does not.
    """
    out: Dict[str, List[str]] = {}
    for lane in C.ANNEAL_RESERVE_LANES:
        ids = sorted(cat.by_lane.get(lane, []))
        if len(ids) >= 2:
            out[lane] = [ids[-1]]
    return out


def compile_timeline(cat: Catalog, total_steps: int, seqs_per_step: int,
                     seq_len: int) -> Timeline:
    alloc = plan.stage_step_allocation(total_steps)
    reserved = choose_reserved_shards(cat)

    available = {lane: len(cat.by_lane.get(lane, [])) for lane in plan.LANES}
    quotas: List[StepQuota] = []
    allocator = DeficitAllocator(plan.LANES)

    for step in range(total_steps):
        stage = plan.stage_at_step(step, alloc)
        mix = plan.mix_at_step(step, alloc)
        in_anneal = stage.startswith(C.ANNEAL_STAGE_PREFIX)

        # A lane with no admitted shard cannot be scheduled. Zeroing it here and
        # renormalising is what keeps the batch full: the alternative is a batch
        # with holes in it, which silently changes the effective batch size.
        unavailable = [l for l in plan.LANES if available.get(l, 0) == 0 and mix[l] > 0]
        eff = {l: (0.0 if available.get(l, 0) == 0 else mix[l]) for l in plan.LANES}
        s = sum(eff.values())
        if s > 0:
            eff = {l: v / s for l, v in eff.items()}

        seqs = allocator.allocate(eff, seqs_per_step)
        q = StepQuota(step=step, stage=stage, n_seqs=seqs_per_step,
                      target_mix=mix, seqs=seqs,
                      floor_required=0, floor_delivered=0,
                      unavailable_lanes=unavailable, reserve_unlocked=in_anneal)
        _apply_floor(q, eff)
        allocator.commit(q.seqs)
        quotas.append(q)

    timeline = Timeline(total_steps=total_steps, seqs_per_step=seqs_per_step,
                        seq_len=seq_len, stage_alloc=alloc, quotas=quotas,
                        feasibility=[], reserved_shards=reserved)
    timeline.feasibility = check_feasibility(timeline, cat)
    return timeline


def _apply_floor(q: StepQuota, eff_mix: Dict[str, float]) -> None:
    """Enforce the protected floor by promotion, recording who paid."""
    floor_lanes = [l for l in plan.FLOOR_LANES]
    required = math.ceil(plan.FLOOR_PCT * q.n_seqs - 1e-9)
    q.floor_required = required

    delivered = sum(q.seqs.get(l, 0) for l in floor_lanes)
    if delivered >= required:
        q.floor_delivered = delivered
        return

    # Lanes eligible to give up a sequence: anything not in the floor set that
    # currently holds at least one.
    for _ in range(required - delivered):
        donors = [l for l in plan.LANES
                  if l not in floor_lanes and q.seqs.get(l, 0) > 0]
        if not donors:
            break
        # Take from whoever is furthest ABOVE its own target share right now.
        donor = max(donors, key=lambda l: (q.seqs[l] / q.n_seqs - eff_mix.get(l, 0.0), l))
        # Give to the native-Indic lane furthest BELOW its own target share.
        receiver = min(floor_lanes,
                       key=lambda l: (q.seqs.get(l, 0) / q.n_seqs - eff_mix.get(l, 0.0), l))
        q.seqs[donor] -= 1
        q.seqs[receiver] = q.seqs.get(receiver, 0) + 1
        q.floor_promotions.append(dict(donor=donor, receiver=receiver, seqs=1))

    q.floor_delivered = sum(q.seqs.get(l, 0) for l in floor_lanes)


def check_feasibility(timeline: Timeline, cat: Catalog) -> List[dict]:
    """Can each lane's planned demand be met from the ADMITTED pool?

    Reported per lane with the epoch count it implies, the scaled cap, and the
    fallback Session 5 declared for that lane. This is the "compiler highlights
    lanes that cannot be satisfied" step, and it runs before a single batch is
    built.
    """
    demand_seqs: Dict[str, int] = {}
    for q in timeline.quotas:
        for lane, n in q.seqs.items():
            demand_seqs[lane] = demand_seqs.get(lane, 0) + n

    out: List[dict] = []
    for lane in plan.LANES:
        supply = cat.lane_tokens(lane)
        reserved_ids = timeline.reserved_shards.get(lane, [])
        reserved_tokens = sum(cat.shards[s].token_count for s in reserved_ids)
        demand_tokens = demand_seqs.get(lane, 0) * timeline.seq_len
        cap_plan = plan.MAX_EPOCHS.get(lane, 1.0)
        cap_scaled = cap_plan * C.EPOCH_CAP_SCALE
        epochs = (demand_tokens / supply) if supply else float("inf")

        status = "ok"
        if supply == 0:
            status = "no_supply"
        elif epochs > cap_scaled:
            status = "over_cap"

        out.append(dict(
            lane=lane,
            planned_seqs=demand_seqs.get(lane, 0),
            demand_tokens=demand_tokens,
            supply_tokens=supply,
            reserved_tokens=reserved_tokens,
            n_shards=len(cat.by_lane.get(lane, [])),
            epochs=round(epochs, 4) if supply else None,
            epoch_cap_session5=cap_plan,
            epoch_cap_scaled=round(cap_scaled, 4),
            status=status,
            fallback=C.SCARCITY_FALLBACK.get(lane) if status != "ok" else None,
        ))
    return out
