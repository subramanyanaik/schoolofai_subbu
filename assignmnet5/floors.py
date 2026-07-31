"""
floors.py — the protected always-on floor, the anneal reserve, and fertility.

FLOOR
  V4 ran an 8% always-on Indic channel. We keep 8% but tighten the DEFINITION:
  only NATIVE Indic (verified + unverified) counts. If translated and synthetic
  counted, an OPUS-class selector could satisfy "8% Indic every batch" entirely
  with cheap generated text and never touch the scarce native tiers -- which is
  precisely the language erasure the floor exists to prevent.

RESERVE
  The anneal reserve is HELD BACK from the lane budgets, not added on top.
  Critically, the verified-native component of the anneal is the SECOND EPOCH
  of verified native, deferred -- not extra exposure. Getting this wrong
  silently pushes the scarcest tier to 3x epochs.
"""
from . import config as C


def floor_report(rows):
    by = {r["lane"]: r for r in rows}
    seqs = round(C.FLOOR["pct"] * C.FLOOR["batch_seqs"])
    native_tokens = sum(by[l]["target"] for l in C.FLOOR["lanes"])
    native_share = native_tokens / C.TRAIN_BUDGET
    # the floor must be satisfiable in EVERY stage, so compare against the
    # stage with the LOWEST native-Indic share
    from .curriculum import STAGE_MIX
    per_stage = {s: sum(m.get(l, 0) for l in C.FLOOR["lanes"])
                 for s, m in STAGE_MIX.items()}
    binding = min(per_stage.items(), key=lambda kv: kv[1])
    return dict(
        pct=C.FLOOR["pct"], seqs_per_batch=seqs, batch_seqs=C.FLOOR["batch_seqs"],
        lanes=C.FLOOR["lanes"], native_global_share=native_share,
        per_stage=per_stage, binding_stage=binding[0], binding_share=binding[1],
        satisfiable=binding[1] >= C.FLOOR["pct"],
        slack=binding[1] - C.FLOOR["pct"],
    )


# Anneal reserve composition. Shares sum to 1.0 of ANNEAL_RESERVE.
RESERVE = [
    dict(name="Instruction-dense general text", share=.28,
         why="matches downstream SFT distribution; sharpens instruction-following late"),
    dict(name="Verified-native Indic (DEFERRED 2nd epoch)", share=.22,
         why="the 2nd pass over verified native is SPENT here, not added; "
             "total exposure stays at 2.0x"),
    dict(name="Answer-verified math / worked solutions", share=.18,
         why="OLMo 2 precedent: small late reserve moved GSM-style 24%->67%"),
    dict(name="Execution-verified agentic trajectories", share=.16,
         why="only sandbox-successful traces; loss on model tokens only"),
    dict(name="Long-CoT Band 3-4 reasoning", share=.11,
         why="effort-dial supervision lands where LR is lowest"),
    dict(name="Safety + long-context refresh", share=.05,
         why="12+ lang refusals incl. romanized attacks; one 256K pass so "
             "long-context survives the cooldown"),
]
# Canary strings live ONLY in private eval suites -- never in the anneal.
RESERVE_EXCLUSIONS = ["benchmark test items", "eval-adjacent paraphrases",
                      "canary strings", "unverified synthetic CoT"]


def reserve_report():
    tot = C.ANNEAL_RESERVE
    rows = [dict(**r, tokens=r["share"] * tot) for r in RESERVE]
    return dict(total=tot, pct_of_budget=tot / C.TRAIN_BUDGET,
                rows=rows, share_sum=sum(r["share"] for r in RESERVE),
                exclusions=RESERVE_EXCLUSIONS)


def verified_native_exposure(rows):
    """The bug-check: total epochs on the scarcest tier, INCLUDING the anneal."""
    by = {r["lane"]: r for r in rows}
    unique = by["indic_verified"]["unique"]
    bulk = by["indic_verified"]["target"]
    anneal_vn = next(r["share"] for r in RESERVE if r["name"].startswith("Verified-native"))
    anneal_tokens = anneal_vn * C.ANNEAL_RESERVE
    return dict(
        unique=unique, bulk_tokens=bulk, anneal_tokens=anneal_tokens,
        # the anneal portion is CARVED OUT of the lane budget, not added
        total_exposure=bulk, epochs=bulk / unique,
        anneal_is_subset=anneal_tokens <= bulk,
        cap=C.MAX_EPOCHS["indic_verified"],
        ok=(bulk / unique) <= C.MAX_EPOCHS["indic_verified"] + 1e-9,
    )


# ---------------------------------------------------------------------------
# Fertility
# ---------------------------------------------------------------------------
def weighted_indic_fertility():
    w = C.LANG_WEIGHTS
    tot = sum(w.values())
    num = sum(w[l] * C.FERTILITY_TARGETS[l] for l in w)
    return num / tot


def effective_words(indic_tokens):
    """What the same Indic token budget BUYS in words, vs baselines.
    Fertility = tokens/word, so words = tokens / fertility."""
    ours = weighted_indic_fertility()
    out = {"ERAV5 target": dict(fertility=ours, words=indic_tokens / ours)}
    for name, f in C.BASELINE_FERTILITY.items():
        out[name] = dict(fertility=f, words=indic_tokens / f)
    base = out["english_centric_bpe"]["words"]
    for v in out.values():
        v["vs_english_bpe"] = v["words"] / base
    return out


def session2_balance_score():
    """The Session-2 assignment metric: Score = 1000 / (Xmax - Xmin).
    Rewards BALANCED cross-lingual tokenization, not just low average."""
    xs = list(C.FERTILITY_TARGETS.values())
    spread = max(xs) - min(xs)
    return dict(x_max=max(xs), x_min=min(xs), spread=spread,
                score=1000.0 / spread if spread else float("inf"))
