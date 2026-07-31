"""
sensitivity.py — what breaks first if the inventory estimate is wrong,
and where the remaining cleaning effort should therefore be aimed.

Two jobs:
  A) SENSITIVITY. The plan rests on estimates. The 90B verified-native figure
     is the most fragile input in the whole document -- every other lane has
     slack. We re-solve the mixture under pessimistic supply and report what
     bends.
  B) CLEANING PRIORITY. The assignment says cleaning now targets the slots the
     mixture shows to be starved. We compute, per lane, how many RAW tokens
     must enter the Session-4 pipeline to yield the clean tokens the plan needs.
"""
import copy
from . import config as C
from . import mixture as M
from . import floors as F


# ---------------------------------------------------------------------------
# A. SENSITIVITY
# ---------------------------------------------------------------------------
SCENARIOS = [
    dict(name="Baseline", mult={}),
    dict(name="Verified native -33% (audit finds more MT)",
         mult={"indic_verified": 0.67}),
    dict(name="Verified native -50% (worst credible)",
         mult={"indic_verified": 0.50}),
    dict(name="Code -30% after license filter",
         mult={"code": 0.70}),
    dict(name="Unverified Indic -40% (lang-ID mislabels)",
         mult={"indic_unverified": 0.60}),
    dict(name="All Indic native -40% (systemic)",
         mult={"indic_verified": 0.60, "indic_unverified": 0.60}),
]


def _solve_with(mult):
    inv = copy.deepcopy(C.INVENTORY)
    for d in inv:
        if d["unique"] is not None and d["lane"] in mult:
            d["unique"] *= mult[d["lane"]]
    orig = C.INVENTORY
    C.INVENTORY = inv
    try:
        rows = M.solve()
        ind = M.indic_rollup(rows)
        fl = F.floor_report(rows)
        t = M.totals(rows)
        vn = F.verified_native_exposure(rows)
    finally:
        C.INVENTORY = orig
    return rows, ind, fl, t, vn


def sensitivity_table():
    out = []
    for sc in SCENARIOS:
        rows, ind, fl, t, vn = _solve_with(sc["mult"])
        by = {r["lane"]: r for r in rows}
        # The TARGET shares are fixed by the mixture; what moves under a supply
        # shock is how much of each target is backed by REAL tokens. Reporting
        # target-based percentages would hide the whole effect.
        real_native = sum(by[l]["real"] for l in C.NATIVE_INDIC_LANES)
        indic_total = ind["total"]
        gap = sum(by[l]["synthetic"] for l in C.INDIC_LANES)
        out.append(dict(
            name=sc["name"],
            vn_epochs=by["indic_verified"]["epochs"],
            vn_gap=by["indic_verified"]["synthetic"],
            real_native=real_native,
            real_native_pct=real_native / indic_total,
            indic_gap=gap,
            code_epochs=by["code"]["epochs"],
            code_gap=by["code"]["synthetic"],
            total_real_pct=1.0 - t["synth_pct"],
            floor_ok=fl["satisfiable"],
            # parity is measured against REAL native, which is the honest test
            syn_to_real_native=(ind["tiers"]["indic_synthetic"]["tokens"] + gap) / real_native
                                if real_native else float("inf"),
        ))
    return out


RESPONSES = {
    "indic_verified": (
        "Hold total exposure at 2.0x (do NOT raise epochs). Backfill the Indic lane "
        "from unverified native -- it has 0.45T unique at only 0.62 epochs, i.e. "
        "~0.17T of unused headroom. Synthetic does NOT expand to fill the gap: the "
        "parity cap is a RATIO to native, so if native shrinks the synthetic ceiling "
        "shrinks with it automatically."),
    "code": (
        "Move code to 1.3-1.5x epochs before considering synthetic code. Repetition "
        "is safer than generation in a lane where SWE-bench leakage is the dominant "
        "risk and where generated code has no independent correctness signal beyond "
        "the tests we also generated."),
    "indic_unverified": (
        "This is the lane with the most slack, so a 40% cut is absorbable at the "
        "current 0.62 epochs. If it and verified BOTH shrink, the Indic headline "
        "share drops from 20% to whatever native supports -- the share follows the "
        "supply, not the other way round."),
    "agentic": (
        "No change. The lane was designed around self-play from the start; P3 gates "
        "whether self-play works at all, which is the real risk here, not supply."),
}


# ---------------------------------------------------------------------------
# B. CLEANING PRIORITY  (Session-4 pipeline yields)
# ---------------------------------------------------------------------------
# Survival rate through the Session-4 pipeline, per lane. These are the
# cumulative multipliers across: extraction -> normalization -> quality
# filter -> dedup -> lang-ID -> PII -> decontamination.
PIPELINE_YIELD = {
    "english":          0.22,   # aggressive edu-classifier filtering
    "code":             0.31,   # license + secret + autogen + near-clone dedup
    "math":             0.18,   # classifier-mined from general web
    "indic_verified":   0.09,   # OCR gates + native audit reject hard
    "indic_unverified": 0.26,   # script-aware thresholds, code-mix kept
    "indic_translated": 0.55,   # already-clean parallel corpora
    "forums":           0.14,   # spam/boilerplate heavy
    "multilingual":     0.24,
    "agentic":          0.42,   # trajectory success gate
    "reasoning":        0.35,   # answer-verification gate
}


def cleaning_plan(rows):
    """How many RAW tokens must enter the pipeline per lane, and which lanes
    the remaining cleaning effort should prioritise."""
    out = []
    for r in rows:
        lane = r["lane"]
        y = PIPELINE_YIELD.get(lane, 0.25)
        # only real (non-generated) tokens need to be cleaned from raw crawl
        need_clean = r["real"]
        # plus the selector headroom: we collect 2x what we train on
        need_collected = need_clean * (C.COLLECTION_TARGET / C.TRAIN_BUDGET)
        raw = need_collected / y
        deficit = max(0.0, r["target"] - r["real"])
        out.append(dict(
            lane=lane, yield_rate=y, clean_needed=need_clean,
            collected_target=need_collected, raw_required=raw,
            generation_deficit=deficit,
            starved=(r["real"] / r["target"] < 0.5) if r["target"] else False,
        ))
    out.sort(key=lambda d: -d["raw_required"])
    return out


def priority_queue(rows):
    """Rank lanes by marginal value of one more cleaned token.
    Scarcity (1/headroom) x strategic weight (floor-eligible lanes count double)."""
    ranked = []
    for r in rows:
        # Lanes with NO real corpus are generated by design -- more cleaning
        # cannot help them, so they are out of scope for this queue entirely.
        if r["target"] == 0 or r["unique"] == 0:
            continue
        headroom = max(r["headroom"], 1e-3)
        scarcity = min(1.0 / headroom, 10.0)          # clamp: no divide-by-zero drama
        weight = 2.0 if r["lane"] in C.FLOOR["lanes"] else 1.0
        ranked.append(dict(lane=r["lane"], score=scarcity * weight,
                           headroom=headroom, epochs=r["epochs"],
                           floor_lane=r["lane"] in C.FLOOR["lanes"]))
    ranked.sort(key=lambda d: -d["score"])
    return ranked
