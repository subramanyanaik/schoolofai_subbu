"""
plan.py — the Session-5 plan, imported rather than retyped.

Session 6's job is to *execute* Session 5's mixture. If the executing system
carried its own private copy of the lane shares, curriculum stages and
protected floor, the two could drift apart silently and every "mixture
compliance" claim downstream would be measuring the wrong target. So this
module imports `assignment5/src/erav5` directly and scales it to the size of
the demo run.

What is imported (not re-declared here):
    * the 11 capability lanes and their global shares      -> erav5.config.MIXTURE
    * the 5 curriculum stages and their token fractions    -> erav5.curriculum.STAGES
    * the per-stage lane mixes, with S1 solved so the      -> erav5.curriculum.STAGE_MIX
      schedule integrates exactly to the global mixture
    * the protected floor: 8% of every batch, native       -> erav5.config.FLOOR
      Indic only
    * per-lane epoch caps                                  -> erav5.config.MAX_EPOCHS

What is added here, because it is an execution concern Session 5 had no reason
to have: the mapping from token-fraction stages onto an integer number of
optimizer steps, and the ramp bands expressed in steps.
"""
from __future__ import annotations

import sys
from typing import Dict, List

from .config import REPO

# --- import the Session-5 plan ---------------------------------------------
_A5_SRC = REPO / "assignment5" / "src"
if not (_A5_SRC / "erav5" / "config.py").exists():
    raise RuntimeError(
        f"Session-5 plan not found at {_A5_SRC}. Session 6 executes Session 5's "
        "mixture and refuses to run against a private copy of it."
    )
if str(_A5_SRC) not in sys.path:
    sys.path.insert(0, str(_A5_SRC))

from erav5 import config as C5           # noqa: E402
from erav5 import curriculum as CUR5     # noqa: E402

# --- re-export the plan -----------------------------------------------------
LANES: List[str] = list(C5.MIXTURE.keys())
GLOBAL_MIXTURE: Dict[str, float] = dict(C5.MIXTURE)
MAX_EPOCHS: Dict[str, float] = dict(C5.MAX_EPOCHS)

FLOOR_PCT: float = C5.FLOOR["pct"]
FLOOR_LANES: List[str] = list(C5.FLOOR["lanes"])
INDIC_LANES: List[str] = list(C5.INDIC_LANES)

STAGE_NAMES: List[str] = [s["name"] for s in CUR5.STAGES]
STAGE_FRACS: Dict[str, float] = {s["name"]: s["frac"] for s in CUR5.STAGES}
STAGE_CTX: Dict[str, int] = {s["name"]: s["ctx"] for s in CUR5.STAGES}
STAGE_NOTE: Dict[str, str] = {s["name"]: s["note"] for s in CUR5.STAGES}
STAGE_MIX: Dict[str, Dict[str, float]] = {
    name: {lane: float(CUR5.STAGE_MIX[name].get(lane, 0.0)) for lane in LANES}
    for name in STAGE_NAMES
}

MAX_PP_PER_100B: float = CUR5.MAX_PP_PER_100B

# Lane -> license tier / provenance, taken from the Session-5 inventory. Used to
# stamp corpus documents with a provenance story that matches the plan.
LANE_LICENSE: Dict[str, str] = {}
LANE_TIER: Dict[str, str] = {}
for _item in C5.INVENTORY:
    LANE_LICENSE.setdefault(_item["lane"], _item["license"])
    LANE_TIER.setdefault(_item["lane"], _item["tier"])

# Minimum steps per curriculum stage. Session 5's S4 anneal is 5% of a 4T-token
# budget; at 24 optimizer steps that rounds to zero, and a stage that never runs
# cannot be audited. Every stage therefore gets at least this many steps, so all
# five stages and all four transitions are actually exercised.
MIN_STEPS_PER_STAGE = 2

# Ramp band straddling each stage boundary, in steps (half in the outgoing
# tail, half in the incoming head -- the straddle rule from Session 5 section 4.2).
# Session 5 sizes these in tokens (30B-142B); at demo scale the token figure is
# sub-step, so the *mechanism* is preserved at the smallest width that makes the
# interpolation observable, and the source token width is carried alongside it
# in the compiled timeline for audit.
RAMP_STEPS = 2


def stage_step_allocation(total_steps: int) -> List[dict]:
    """Split `total_steps` across the five Session-5 stages.

    Largest-remainder over the Session-5 token fractions, after reserving
    MIN_STEPS_PER_STAGE for each stage. Largest remainder (rather than rounding
    each independently) guarantees the parts sum to exactly `total_steps`.
    """
    n = len(STAGE_NAMES)
    reserved = MIN_STEPS_PER_STAGE * n
    if total_steps < reserved:
        raise ValueError(f"need >= {reserved} steps to give all {n} stages a turn")

    free = total_steps - reserved
    exact = {name: STAGE_FRACS[name] * free for name in STAGE_NAMES}
    floors = {name: int(exact[name]) for name in STAGE_NAMES}
    remainder = free - sum(floors.values())
    # Ties broken by stage order so the allocation is deterministic.
    order = sorted(STAGE_NAMES, key=lambda s: (-(exact[s] - floors[s]), STAGE_NAMES.index(s)))
    for name in order[:remainder]:
        floors[name] += 1

    out: List[dict] = []
    cursor = 0
    for name in STAGE_NAMES:
        steps = floors[name] + MIN_STEPS_PER_STAGE
        out.append(dict(
            stage=name,
            start_step=cursor,
            end_step=cursor + steps,      # exclusive
            steps=steps,
            plan_frac=STAGE_FRACS[name],
            plan_ctx=STAGE_CTX[name],
            note=STAGE_NOTE[name],
        ))
        cursor += steps
    assert cursor == total_steps
    return out


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix_at_step(step: int, alloc: List[dict]) -> Dict[str, float]:
    """Lane shares in effect at `step`, with stage transitions ramped.

    Session 5's rule: lane shares are never stepped, they are ramped across a
    band that straddles the boundary. Inside the band the two adjacent stage
    mixes are linearly interpolated, so no lane's share jumps -- which is the
    whole point of the band (V4 recorded a ~150x gradient-norm spike from an
    abrupt share change; the Session-5 P4 proxy reproduced 7.04x on real text
    and showed it collapsing to ~1.8x with even a short ramp).
    """
    idx = _stage_index(step, alloc)
    cur = alloc[idx]
    mix = STAGE_MIX[cur["stage"]]
    half = RAMP_STEPS // 2

    # Head of the band: blending IN from the previous stage.
    if idx > 0 and step - cur["start_step"] < half:
        prev = STAGE_MIX[alloc[idx - 1]["stage"]]
        # position 0 of the head is the midpoint of the ramp -> t = 0.5
        offset = step - cur["start_step"]
        t = 0.5 + (offset + 1) / (2.0 * half + 1e-12) * 0.5
        return {l: _lerp(prev[l], mix[l], min(t, 1.0)) for l in LANES}

    # Tail of the band: blending OUT toward the next stage.
    if idx < len(alloc) - 1 and cur["end_step"] - step <= half:
        nxt = STAGE_MIX[alloc[idx + 1]["stage"]]
        remaining = cur["end_step"] - step        # 1..half
        t = (half - remaining + 1) / (2.0 * half + 1e-12) * 0.5
        return {l: _lerp(mix[l], nxt[l], min(max(t, 0.0), 0.5)) for l in LANES}

    return dict(mix)


def _stage_index(step: int, alloc: List[dict]) -> int:
    for i, a in enumerate(alloc):
        if a["start_step"] <= step < a["end_step"]:
            return i
    return len(alloc) - 1


def stage_at_step(step: int, alloc: List[dict]) -> str:
    return alloc[_stage_index(step, alloc)]["stage"]


def transition_report() -> List[dict]:
    """Session 5's ramp bands, carried through for the audit record."""
    return CUR5.blend_bands()
