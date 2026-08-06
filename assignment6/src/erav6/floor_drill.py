"""
floor_drill.py — a controlled experiment: what is the protected floor for?

The main run does not exercise the protected-floor override, and the reason is
worth stating plainly rather than engineering around. This system's proxy
utility is loss headroom measured from the live model, and with a byte-level
tokenizer Devanagari and Tamil are the HIGHEST-loss content in the corpus.
Measured over the real run: median utility +1.22 for indic_verified and +1.20
for indic_unverified, against a global median of -0.54. The selector prefers
native Indic. It never needs rescuing, so the override never fires.

That is a genuine result, not a gap to paper over. But it leaves the floor
untested, and Session 5's argument for the floor is about a DIFFERENT kind of
proxy: "English-tuned quality proxies systematically reject Indic script,
code-mixing, and local content, so an unprotected selector starves the lane."

So this module builds exactly that proxy and runs it, as a clearly-labelled
counterfactual next to the real run:

    arm A  UNPROTECTED  English-tuned proxy, selector free to take the top-N
                        candidates by score with no floor and no lane quota
    arm B  PROTECTED    same proxy, same candidates, same step -- but with the
                        compiled lane quotas and the protected-floor override

Both arms score the SAME candidate pool with the SAME model, so the only
difference is the floor. The bias is applied through `OpusSelector(lane_bias=)`,
which stamps `-lane-biased-drill` into the proxy version, so a drill decision
can never be confused with a real one in the ledger.

This is Session 5's P2 proxy ("protected floor necessity") in miniature: it
cannot tell us what the floor is worth in MILU points, but it does show
mechanically what an unprotected selector does to the lane when the proxy is
the kind Session 5 was worried about.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from . import config as C
from . import plan
from .catalog import Catalog
from .dataloader import DataLoader
from .mixture import Timeline
from .opus import Candidate, OpusSelector

# The English-tuned proxy Session 5 describes: it marks down Indic script and
# code-mixed text regardless of how much the model still has to learn from it.
#
# The magnitude is chosen deliberately. Utility z-scores in the real run span
# roughly -2.0 to +3.4, so a penalty of -3.0 leaves native Indic sitting ABOVE
# the pool average (measured: -0.39 biased, against -0.58 for everything else)
# and the unprotected arm still picks it up. That is not the proxy Session 5 is
# describing -- a proxy that "systematically rejects Indic script" has to
# outweigh the evidence, not tie with it. The penalty is therefore set larger
# than the full observed spread of the headroom score, which is what makes this
# a model of rejection rather than of mild dispreference.
ENGLISH_TUNED_BIAS: Dict[str, float] = {
    "indic_verified": -8.0,
    "indic_unverified": -8.0,
    "indic_translated": -8.0,
    "indic_synthetic": -8.0,
    "forums": -4.0,          # romanized / code-mixed
    "multilingual": -2.5,
}


def run(cat: Catalog, timeline: Timeline, profile: dict, model,
        device: torch.device, step: int) -> dict:
    """Score one step's candidates twice: without the floor, then with it."""
    quota = timeline.at(step)
    seqs_per_step = profile["n_ranks"] * profile["microbatch"] * profile["grad_accum"]

    # One candidate pool, built once, shared by both arms.
    selector_a = OpusSelector(lane_bias=ENGLISH_TUNED_BIAS)
    loader = DataLoader(cat, timeline, profile, "drill", C.DATA_SEED, selector_a)
    candidates: List[Candidate] = []
    for lane in sorted(quota.seqs):
        if quota.seqs[lane] <= 0:
            continue
        cands, _ = loader._candidates_for_lane(
            lane, quota.seqs[lane], quota.reserve_unlocked, step)
        candidates.extend(cands)

    if not candidates:
        return dict(ok=False, reason="no candidates available for the drill")

    headroom = selector_a.measure_headroom(candidates, model, device)
    import statistics
    mu = statistics.fmean(headroom)
    sigma = statistics.pstdev(headroom) if len(headroom) > 1 else 0.0

    scored = []
    for cand, raw in zip(candidates, headroom):
        z = (raw - mu) / sigma if sigma > 1e-9 else 0.0
        fit = OpusSelector.stage_fit_of(cand.lane, quota.stage)
        biased = (z + C.OPUS["stage_fit_weight"] * fit
                  + ENGLISH_TUNED_BIAS.get(cand.lane, 0.0))
        unbiased = z + C.OPUS["stage_fit_weight"] * fit
        scored.append(dict(cand=cand, lane=cand.lane, raw=raw,
                           biased=biased, unbiased=unbiased))

    floor_lanes = set(plan.FLOOR_LANES)

    # -- arm A: unprotected. Top-N by the biased score, no quota, no floor. ---
    arm_a = sorted(scored, key=lambda s: (-s["biased"], s["cand"].candidate_id))
    a_take = arm_a[:seqs_per_step]
    a_floor = sum(1 for s in a_take if s["lane"] in floor_lanes)
    a_lanes: Dict[str, int] = {}
    for s in a_take:
        a_lanes[s["lane"]] = a_lanes.get(s["lane"], 0) + 1

    # -- arm B: protected. Compiled quotas plus the floor override. ----------
    selector_b = OpusSelector(lane_bias=ENGLISH_TUNED_BIAS)
    selector_b.recent = []
    accepted, decisions = selector_b.select(
        candidates, dict(quota.seqs), step, quota.stage, model, device,
        model_checkpoint="drill")
    accepted, decisions = loader._enforce_floor(
        accepted, decisions, candidates, quota.floor_required, step)
    b_floor = sum(1 for c in accepted if c.lane in floor_lanes)
    b_lanes: Dict[str, int] = {}
    for c in accepted:
        b_lanes[c.lane] = b_lanes.get(c.lane, 0) + 1
    overrides = [d for d in decisions if d.protected_floor_override]

    native_scores = [s["biased"] for s in scored if s["lane"] in floor_lanes]
    other_scores = [s["biased"] for s in scored if s["lane"] not in floor_lanes]

    return dict(
        ok=True,
        step=step,
        stage=quota.stage,
        proxy="english_tuned_quality_proxy",
        proxy_version=selector_b.proxy_version,
        lane_bias=ENGLISH_TUNED_BIAS,
        n_candidates=len(candidates),
        floor_required=quota.floor_required,
        floor_lanes=sorted(floor_lanes),
        unprotected=dict(
            native_indic_sequences=a_floor,
            native_indic_share=round(a_floor / max(1, len(a_take)), 6),
            lanes=dict(sorted(a_lanes.items())),
            floor_met=a_floor >= quota.floor_required,
        ),
        protected=dict(
            native_indic_sequences=b_floor,
            native_indic_share=round(b_floor / max(1, len(accepted)), 6),
            lanes=dict(sorted(b_lanes.items())),
            floor_met=b_floor >= quota.floor_required,
            n_floor_overrides=len(overrides),
            overrides=[d.as_dict() for d in overrides],
        ),
        biased_score_gap=dict(
            mean_native_indic=round(sum(native_scores) / max(1, len(native_scores)), 6),
            mean_other=round(sum(other_scores) / max(1, len(other_scores)), 6),
        ),
        conclusion=(
            "the unprotected selector starves the native-Indic lane under an "
            "English-tuned proxy; the floor restores it via explicit override "
            "records" if a_floor < quota.floor_required <= b_floor else
            "the floor made no difference for this candidate pool"),
        floor_was_necessary=a_floor < quota.floor_required <= b_floor,
    )
