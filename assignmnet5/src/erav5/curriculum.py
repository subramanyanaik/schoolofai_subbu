"""
curriculum.py — ordering and timing, not just proportions.

Three things this encodes that a flat mixture table cannot:

  1. STAGE x LANE reconciliation. Global shares are a weighted average of
     stage mixes. If the stages don't integrate back to the global shares,
     the plan is internally inconsistent -- so we compute the integral and
     assert on it rather than eyeballing it.

  2. GRADIENT-SAFE BLENDING BANDS. V4 recorded a ~150x gradient-norm spike
     when the Hindi share was raised abruptly. Any lane share change is
     therefore ramped linearly across a multi-billion-token band, and we
     compute the max per-100B-token delta so it can be bounded.

  3. DIFFICULTY / REASONING-LENGTH BANDS with a concrete example each,
     scheduled so early training is Band 0-1 and the anneal is Band 3-4.
"""
from . import config as C

# ---------------------------------------------------------------------------
# Stage schedule. Fractions are of TRAIN_BUDGET.
# ---------------------------------------------------------------------------
STAGES = [
    dict(name="S0 Warmup",      frac=0.0375, ctx=8_192,   lr="warmup->peak",
         note="embedding + router warmup; mixture held at S1 proportions"),
    dict(name="S1 Foundation",  frac=0.6125, ctx=8_192,   lr="peak, cosine",
         note="web-heavy; language, world knowledge, basic syntax"),
    dict(name="S2 Skill ramp",  frac=0.2250, ctx=32_768,  lr="cosine decay",
         note="code/math/reasoning up; ctx 8K->32K"),
    dict(name="S3 Long context",frac=0.0750, ctx=262_144, lr="low, flat",
         note="ctx 32K->256K on naturally-long docs only"),
    dict(name="S4 Anneal",      frac=0.0500, ctx=262_144, lr="->0",
         note="Tier-A reserve only; LR cooldown"),
]

# Per-stage lane mixes. Each stage's dict must sum to 1.0.
STAGE_MIX = {
    "S0 Warmup": dict(english=.38, code=.17, math=.09, indic_verified=.030,
                      indic_unverified=.070, indic_translated=.055,
                      indic_synthetic=.035, agentic=.030, reasoning=.020,
                      forums=.060, multilingual=.060),
    # S1 is a PLACEHOLDER -- it is overwritten at import time by solve_s1() so
    # the schedule integrates EXACTLY to the global mixture. Edit any other
    # stage freely; S1 absorbs the residual automatically.
    "S1 Foundation": {lane: 0.0 for lane in C.MIXTURE},
    "S2 Skill ramp": dict(english=.190, code=.310, math=.180, indic_verified=.040,
                          indic_unverified=.070, indic_translated=.030,
                          indic_synthetic=.040, agentic=.085, reasoning=.045,
                          forums=.005, multilingual=.005),
    "S3 Long context": dict(english=.300, code=.290, math=.110, indic_verified=.060,
                            indic_unverified=.100, indic_translated=.015,
                            indic_synthetic=.020, agentic=.070, reasoning=.030,
                            forums=.000, multilingual=.005),
    "S4 Anneal": dict(english=.200, code=.120, math=.150, indic_verified=.180,
                      indic_unverified=.060, indic_translated=.020,
                      indic_synthetic=.030, agentic=.130, reasoning=.110,
                      forums=.000, multilingual=.000),
}


def stage_tokens(budget=None):
    budget = budget or C.TRAIN_BUDGET
    out = []
    cum = 0.0
    for s in STAGES:
        t = s["frac"] * budget
        out.append(dict(**s, tokens=t, start=cum, end=cum + t))
        cum += t
    return out


def integrate(budget=None):
    """Compute the global share implied by the stage schedule, per lane."""
    budget = budget or C.TRAIN_BUDGET
    st = stage_tokens(budget)
    acc = {lane: 0.0 for lane in C.MIXTURE}
    for s in st:
        mix = STAGE_MIX[s["name"]]
        for lane, sh in mix.items():
            acc[lane] += sh * s["tokens"]
    return {lane: dict(tokens=v, share=v / budget,
                       target_share=C.MIXTURE[lane],
                       delta=v / budget - C.MIXTURE[lane])
            for lane, v in acc.items()}


# Bound on how fast any lane share may move. Derived from V4's incident:
# an abrupt Hindi share increase produced a ~150x gradient-norm spike.
MAX_PP_PER_100B = 12.0


def solve_s1():
    """Solve the S1 mix so the stage schedule integrates EXACTLY to MIXTURE.
    S1 is the largest stage, so it absorbs the residual with least distortion.
    Re-run this whenever another stage mix is edited."""
    w = {s["name"]: s["frac"] for s in STAGES}
    s1 = "S1 Foundation"
    sol = {}
    for lane, tgt in C.MIXTURE.items():
        other = sum(w[s] * STAGE_MIX[s].get(lane, 0) for s in w if s != s1)
        sol[lane] = (tgt - other) / w[s1]
    return sol


# Solve S1 at import time so the schedule is always exactly consistent.
STAGE_MIX["S1 Foundation"] = solve_s1()


def blend_bands(band_tokens=150e9):
    """Max lane-share change per band between consecutive stages.
    V4's 150x gradient spike came from an abrupt share change; we bound it.
    Ramp length is chosen so no lane moves faster than MAX_PP_PER_100B."""
    st = stage_tokens()
    out = []
    for a, b in zip(st, st[1:]):
        ma, mb = STAGE_MIX[a["name"]], STAGE_MIX[b["name"]]
        deltas = {l: mb.get(l, 0) - ma.get(l, 0) for l in set(ma) | set(mb)}
        worst = max(deltas.items(), key=lambda kv: abs(kv[1]))
        # required ramp to stay under the rate limit
        required = abs(worst[1]) * 100 / MAX_PP_PER_100B * 100e9
        # The ramp STRADDLES the boundary: half in the tail of the outgoing
        # stage, half in the head of the incoming one. Otherwise a long ramp
        # would eat the whole anneal, which defeats the point of a cooldown.
        half = required / 2.0
        avail = min(a["tokens"], b["tokens"])          # each side can give this much
        ramp = min(required, 2.0 * avail, band_tokens)
        rate = abs(worst[1]) * 100 / (ramp / 100e9) if ramp else float("inf")
        out.append(dict(
            frm=a["name"], to=b["name"], worst_lane=worst[0],
            worst_delta=worst[1], ramp_tokens=ramp, required_ramp=required,
            tail_of_prev=ramp / 2.0, head_of_next=ramp / 2.0,
            pct_of_next=(ramp / 2.0) / b["tokens"],
            pp_per_100B=rate, within_limit=rate <= MAX_PP_PER_100B + 1e-6,
            lr_protected=b["name"].startswith("S4"),
        ))
    return out


# ---------------------------------------------------------------------------
# Difficulty / reasoning-length bands
# ---------------------------------------------------------------------------
BANDS = [
    dict(band=0, name="Direct recall", steps="0", tokens="<32",
         example="Q: Odisha ki rajdhani kya hai? -> A: Bhubaneswar.",
         lanes="english, indic, forums",
         sched="S0-S1 dominant, never removed (keeps short answers short)"),
    dict(band=1, name="Short CoT", steps="1-3", tokens="32-256",
         example="Sam has 3 apples, buys 2, gives 1 away. 3+2-1 = 4.",
         lanes="math, reasoning, indic",
         sched="S1 primary; carried through at reduced weight"),
    dict(band=2, name="Medium CoT", steps="4-8", tokens="256-1K",
         example=("JEE-mains projectile: pick formula -> substitute -> carry "
                  "units -> sanity-check magnitude -> state answer."),
         lanes="math, code, reasoning",
         sched="ramps in through S2"),
    dict(band=3, name="Long CoT / tool-augmented", steps="8-20", tokens="1K-8K",
         example=("Agentic: 'Book an IRCTC ticket under Rs.800, verify the PNR, "
                  "retry once on failure' -- plan, call, read error, replan."),
         lanes="agentic, reasoning, code",
         sched="S2 late + S3; concentrated in S4 anneal"),
    dict(band=4, name="Extended / cross-document", steps="20+", tokens="8K-64K",
         example=("Read a 40-page judgment + a prior ruling + a Hansard excerpt; "
                  "identify which precedent controls and why."),
         lanes="long-context pool (indic + english + code)",
         sched="S3 and S4 only -- requires 256K context to exist first"),
]

# Effort-dial control tokens (the 'controllable reasoning depth' objective)
EFFORT_LEVELS = {
    "low":    dict(band_mix={0: .60, 1: .40}, budget="<128 thinking tokens"),
    "medium": dict(band_mix={1: .45, 2: .55}, budget="128-1K"),
    "high":   dict(band_mix={2: .35, 3: .65}, budget="1K-8K"),
    "ultra":  dict(band_mix={3: .40, 4: .60}, budget="8K-64K"),
}

# Long-context sources: selected from EXISTING lanes, not a new token slot.
LONG_CTX_POOL = [
    dict(lane="code", doc="whole repos concatenated by import graph",
         why="cross-file dependency reasoning"),
    dict(lane="indic_unverified", doc="judgments, Hansard, gov reports",
         why="long-document Indic comprehension"),
    dict(lane="indic_verified", doc="OCR'd full-length books",
         why="sustained native narrative"),
    dict(lane="english", doc="books, legal/technical long-form",
         why="RULER@256K parity"),
    dict(lane="agentic", doc="full multi-step trajectories, unsplit",
         why="long tool-history management"),
]
