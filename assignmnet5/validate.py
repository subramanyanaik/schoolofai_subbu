"""
validate.py — assertions that catch the failure modes this assignment exists
to prevent. Every check prints PASS/FAIL with the actual numbers, so the
plan cannot silently drift out of consistency when someone edits config.py.
"""
from . import config as C
from . import mixture as M
from . import curriculum as CU
from . import floors as F

TOL = 1e-6


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail):
        self.rows.append((name, bool(ok), detail))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def run_all(rows):
    c = Check()
    t = M.totals(rows)
    ind = M.indic_rollup(rows)

    # --- arithmetic integrity ------------------------------------------------
    c.add("Mixture shares sum to 1.0", abs(t["share"] - 1.0) < 1e-9,
          f"sum = {t['share']:.6f}")
    c.add("Stage fractions sum to 1.0",
          abs(sum(s['frac'] for s in CU.STAGES) - 1.0) < 1e-9,
          f"sum = {sum(s['frac'] for s in CU.STAGES):.6f}")
    for s, mix in CU.STAGE_MIX.items():
        c.add(f"Stage mix sums to 1.0 :: {s}", abs(sum(mix.values()) - 1.0) < 1e-6,
              f"sum = {sum(mix.values()):.6f}")

    # --- stage x lane integrates back to global shares -----------------------
    integ = CU.integrate()
    worst = max(integ.items(), key=lambda kv: abs(kv[1]["delta"]))
    c.add("Stage schedule integrates to global mixture (<=0.5pp drift)",
          abs(worst[1]["delta"]) <= 0.005,
          f"worst lane '{worst[0]}' drift = {worst[1]['delta']*100:+.2f}pp")

    # --- supply discipline ---------------------------------------------------
    for r in rows:
        if r["unique"] > 0:
            cap = C.MAX_EPOCHS[r["lane"]]
            c.add(f"Epochs within cap :: {r['lane']}", r["epochs"] <= cap + TOL,
                  f"{r['epochs']:.2f}x vs cap {cap:.1f}x")
    for r in rows:
        c.add(f"Epochs below free-repetition ceiling :: {r['lane']}",
              r["epochs"] <= C.REPETITION_FREE_EPOCHS,
              f"{r['epochs']:.2f}x (ceiling {C.REPETITION_FREE_EPOCHS:.0f}x)")

    # --- no silent generation ------------------------------------------------
    for r in rows:
        if r["synthetic"] > 0:
            c.add(f"Synthetic gap is declared-generable :: {r['lane']}",
                  r["can_generate"],
                  f"needs {r['synthetic']/1e9:.0f}B generated "
                  f"({r['synth_pct']*100:.0f}% of lane)")

    # --- Indic discipline ----------------------------------------------------
    c.add("Synthetic Indic <= parity with native", ind["parity_ok"],
          f"synthetic/native = {ind['synthetic_to_native']:.2f}x "
          f"(cap {ind['parity_cap']:.1f}x)")
    c.add("Indic lane is majority NATIVE", ind["native_pct"] > 0.50,
          f"native = {ind['native_pct']*100:.1f}% of Indic lane")

    # --- the epoch bug that killed the earlier draft -------------------------
    vn = F.verified_native_exposure(rows)
    c.add("Verified-native anneal is a SUBSET of lane budget, not additive",
          vn["anneal_is_subset"],
          f"anneal {vn['anneal_tokens']/1e9:.0f}B <= lane {vn['bulk_tokens']/1e9:.0f}B")
    c.add("Verified-native TOTAL exposure within cap", vn["ok"],
          f"{vn['epochs']:.2f}x vs cap {vn['cap']:.1f}x")

    # --- floor ---------------------------------------------------------------
    fl = F.floor_report(rows)
    c.add("Protected floor satisfiable in every stage", fl["satisfiable"],
          f"binding stage '{fl['binding_stage']}' has "
          f"{fl['binding_share']*100:.1f}% native vs floor {fl['pct']*100:.0f}%")

    # --- reserve -------------------------------------------------------------
    rr = F.reserve_report()
    c.add("Anneal reserve shares sum to 1.0", abs(rr["share_sum"] - 1.0) < 1e-9,
          f"sum = {rr['share_sum']:.4f}")
    anneal_stage = next(s for s in CU.stage_tokens() if s["name"] == "S4 Anneal")
    c.add("Reserve size == S4 stage size", abs(rr["total"] - anneal_stage["tokens"]) < 1e6,
          f"reserve {rr['total']/1e9:.0f}B vs S4 {anneal_stage['tokens']/1e9:.0f}B")

    # --- gradient-safety of mixture transitions ------------------------------
    for bd in CU.blend_bands():
        c.add(f"Blend ramp within rate limit :: {bd['frm'][:2]}->{bd['to'][:2]}",
              bd["within_limit"],
              f"{bd['worst_delta']*100:+.1f}pp on '{bd['worst_lane']}' over "
              f"{bd['ramp_tokens']/1e9:.0f}B = {bd['pp_per_100B']:.1f}pp/100B")
        c.add(f"Ramp fits inside stages :: {bd['frm'][:2]}->{bd['to'][:2]}",
              bd["pct_of_next"] <= 0.50,
              f"head consumes {bd['pct_of_next']*100:.0f}% of incoming stage")

    # --- collection gate -----------------------------------------------------
    c.add("Collection target exceeds train budget (selector has headroom)",
          C.COLLECTION_TARGET > C.TRAIN_BUDGET,
          f"{C.COLLECTION_TARGET/1e12:.1f}T collected vs {C.TRAIN_BUDGET/1e12:.1f}T trained "
          f"({C.COLLECTION_TARGET/C.TRAIN_BUDGET:.1f}x)")

    # --- fertility -----------------------------------------------------------
    wf = F.weighted_indic_fertility()
    c.add("Weighted Indic fertility <= 1.75 target", wf <= 1.75,
          f"weighted = {wf:.4f}")

    return c
