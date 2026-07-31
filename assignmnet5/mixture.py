"""
mixture.py — Turn shares into tokens, then check every lane against REAL supply.

This is the module that catches wishful accounting. For each lane it answers
three questions a reviewer will ask:

  1. How many unique real tokens exist?
  2. How many epochs does the plan run over them?
  3. If that still doesn't fill the share, how much must be GENERATED --
     and is that generation verifiable?

It also implements the Session-5 loss-mapping correction: in agentic
trajectories, tool observations are context-only (zero loss). A lane's
"token share" therefore overstates its real training signal, and the
supervised-token column is the honest one.
"""
from collections import defaultdict
from . import config as C


def lane_supply():
    """Aggregate unique real supply and loss fraction per lane."""
    sup = defaultdict(float)
    loss_w = defaultdict(float)
    generated = defaultdict(bool)
    for d in C.INVENTORY:
        lane = d["lane"]
        if d["unique"] is None:
            generated[lane] = True
            continue
        sup[lane] += d["unique"]
        loss_w[lane] += d["unique"] * d["loss_frac"]
    lf = {k: (loss_w[k] / sup[k] if sup[k] else 1.0) for k in sup}
    # lanes that are purely generated still need a loss fraction
    for d in C.INVENTORY:
        if d["unique"] is None and d["lane"] not in lf:
            lf[d["lane"]] = d["loss_frac"]
    return dict(sup), lf, dict(generated)


def solve(budget=None, mixture=None):
    budget = budget or C.TRAIN_BUDGET
    mixture = mixture or C.MIXTURE
    sup, lossfrac, gen = lane_supply()

    rows = []
    for lane, share in mixture.items():
        target = share * budget
        unique = sup.get(lane, 0.0)
        cap = C.MAX_EPOCHS.get(lane, 1.0)

        # How much can real data cover, respecting the epoch cap?
        real_capacity = unique * cap
        real_used = min(target, real_capacity)
        epochs = (real_used / unique) if unique else 0.0
        synth = target - real_used

        # loss-bearing tokens (Session-5 loss mapping)
        lf = lossfrac.get(lane, 1.0)
        supervised = target * lf

        rows.append(dict(
            lane=lane, share=share, target=target,
            unique=unique, epochs=epochs, real=real_used,
            synthetic=synth, synth_pct=(synth / target if target else 0.0),
            loss_frac=lf, supervised=supervised,
            headroom=(unique / target if target else float("inf")),
            can_generate=gen.get(lane, False),
        ))
    rows.sort(key=lambda r: -r["target"])
    return rows


def totals(rows):
    t = sum(r["target"] for r in rows)
    return dict(
        tokens=t,
        share=sum(r["share"] for r in rows),
        real=sum(r["real"] for r in rows),
        synthetic=sum(r["synthetic"] for r in rows),
        supervised=sum(r["supervised"] for r in rows),
        synth_pct=sum(r["synthetic"] for r in rows) / t,
        supervised_pct=sum(r["supervised"] for r in rows) / t,
    )


def indic_rollup(rows):
    """The four-tier Indic breakdown the assignment demands explicitly."""
    by = {r["lane"]: r for r in rows}
    tiers = {}
    for lane in C.INDIC_LANES:
        r = by[lane]
        tiers[lane] = dict(tokens=r["target"], unique=r["unique"],
                           epochs=r["epochs"], share_of_budget=r["share"])
    total = sum(v["tokens"] for v in tiers.values())
    native = sum(tiers[l]["tokens"] for l in C.NATIVE_INDIC_LANES)
    synth = tiers["indic_synthetic"]["tokens"]
    trans = tiers["indic_translated"]["tokens"]
    for lane, v in tiers.items():
        v["share_of_indic"] = v["tokens"] / total
    return dict(
        tiers=tiers, total=total, native=native,
        non_native=trans + synth,
        native_pct=native / total,
        non_native_pct=(trans + synth) / total,
        synthetic_to_native=synth / native,
        parity_cap=C.SYNTHETIC_PARITY_CAP,
        parity_ok=(synth / native) <= C.SYNTHETIC_PARITY_CAP,
        headline_share=total / C.TRAIN_BUDGET,
    )


def starved_lanes(rows, threshold=0.5):
    """Lanes where real supply covers less than `threshold` of the target.
    These are where the cleaning effort must now be aimed."""
    return [r for r in rows if r["target"] and (r["real"] / r["target"]) < threshold]
