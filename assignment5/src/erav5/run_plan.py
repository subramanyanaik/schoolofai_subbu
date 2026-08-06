"""
run_plan.py — execute the whole plan and print every number.

  python3 -m erav5.run_plan
"""
import sys
from . import config as C
from . import budget as B
from . import mixture as M
from . import curriculum as CU
from . import floors as F
from . import proxy as P
from . import validate as V
from . import sensitivity as S

T = lambda x: f"{x/1e12:.3f}T"
G = lambda x: f"{x/1e9:.1f}B"


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def main():
    # ---------------------------------------------------------------- 1
    hdr("1 · COMPUTE ENVELOPE AND TOKEN BUDGET")
    b = B.budget_report()
    print(f"Flagship            : {b['model']}  "
          f"{b['total_params']/1e9:.0f}B total / {b['active_params']/1e9:.0f}B active "
          f"(sparsity {b['sparsity']*100:.1f}%)")
    print(f"Cluster             : {C.CLUSTER['n_gpus']} GPU x {C.CLUSTER['days']}d "
          f"@ {C.CLUSTER['mfu']*100:.0f}% MFU  ->  {b['cluster_flops']:.3e} FLOPs")
    print(f"Feasible tokens     : {T(b['feasible_tokens'])}  (C / 6N_active)")
    print(f"PLANNED tokens      : {T(b['planned_tokens'])}   "
          f"[compute headroom {b['compute_headroom']:.2f}x]")
    print(f"Chinchilla floor    : {T(b['chinchilla_floor'])} "
          f"(20 tok/active-param) -> we overtrain {b['overtrain_x_chinchilla']:.1f}x")
    print(f"tok / active param  : {b['tok_per_active_param']:.0f}")
    print(f"tok / total param   : {b['tok_per_total_param']:.1f}   "
          f"<- the number that LOOKS small; MoE capacity is fed by diversity, not repeats")
    print(f"vs ERA V4           : {T(b['v4_tokens'])} -> {T(b['planned_tokens'])} "
          f"= {b['v5_over_v4_tokens']:.2f}x the data")
    print(f"Collection target   : {T(b['collection_target'])} cleaned "
          f"= {b['selection_headroom']:.1f}x the train budget "
          f"(so the selector can REJECT, not repeat)")

    print("\n  Model family:")
    print(f"  {'name':<14}{'kind':<7}{'total':>8}{'active':>9}{'tok/act':>9}{'tokens':>10}  role")
    for r in B.variant_table():
        print(f"  {r['name']:<14}{r['kind']:<7}{r['total']/1e9:>7.0f}B"
              f"{r['active']/1e9:>8.0f}B{r['tok_per_active']:>9.0f}"
              f"{T(r['tokens']):>10}  {r['role']}")

    # ---------------------------------------------------------------- 2
    hdr("2 · MIXTURE vs REAL SUPPLY  (the wishful-accounting audit)")
    rows = M.solve()
    print(f"{'lane':<20}{'share':>7}{'tokens':>9}{'unique':>9}{'ep':>6}"
          f"{'real':>9}{'synth':>9}{'syn%':>6}{'loss%':>7}{'sup.tok':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['lane']:<20}{r['share']*100:>6.1f}%{T(r['target']):>9}"
              f"{T(r['unique']):>9}{r['epochs']:>5.2f}x{T(r['real']):>9}"
              f"{T(r['synthetic']):>9}{r['synth_pct']*100:>5.0f}%"
              f"{r['loss_frac']*100:>6.0f}%{T(r['supervised']):>9}")
    t = M.totals(rows)
    print("-" * 78)
    print(f"{'TOTAL':<20}{t['share']*100:>6.1f}%{T(t['tokens']):>9}{'':>9}{'':>6}"
          f"{T(t['real']):>9}{T(t['synthetic']):>9}{t['synth_pct']*100:>5.0f}%"
          f"{t['supervised_pct']*100:>6.0f}%{T(t['supervised']):>9}")
    print(f"\n  Real tokens        : {T(t['real'])} ({(1-t['synth_pct'])*100:.1f}%)")
    print(f"  Generated tokens   : {T(t['synthetic'])} ({t['synth_pct']*100:.1f}%)")
    print(f"  Loss-bearing tokens: {T(t['supervised'])} ({t['supervised_pct']*100:.1f}%)")
    print("    ^ Session-5 loss mapping: tool observations are context-only. A lane's")
    print("      token share overstates its training signal; this column is the honest one.")

    st = M.starved_lanes(rows, 0.5)
    print(f"\n  STARVED LANES (real supply < 50% of target) -> aim the cleaning here:")
    for r in st:
        print(f"    - {r['lane']:<20} real covers {r['real']/r['target']*100:>5.1f}% "
              f"| must generate {T(r['synthetic'])}")

    # ---------------------------------------------------------------- 3
    hdr("3 · INDIC LANE — THE FOUR-TIER SPLIT")
    ind = M.indic_rollup(rows)
    print(f"Indic headline share: {ind['headline_share']*100:.1f}% = {T(ind['total'])}\n")
    print(f"{'tier':<22}{'tokens':>9}{'unique':>9}{'epochs':>8}{'of budget':>11}{'of Indic':>10}")
    print("-" * 78)
    for lane, v in ind["tiers"].items():
        print(f"{lane:<22}{T(v['tokens']):>9}{T(v['unique']):>9}"
              f"{v['epochs']:>7.2f}x{v['share_of_budget']*100:>10.1f}%"
              f"{v['share_of_indic']*100:>9.1f}%")
    print("-" * 78)
    print(f"  NATIVE (verified+unverified) : {T(ind['native'])}  "
          f"= {ind['native_pct']*100:.1f}% of the Indic lane")
    print(f"  NON-NATIVE (trans+synthetic) : {T(ind['non_native'])}  "
          f"= {ind['non_native_pct']*100:.1f}%")
    print(f"  synthetic / native ratio     : {ind['synthetic_to_native']:.2f}x  "
          f"(cap {ind['parity_cap']:.1f}x) -> {'OK' if ind['parity_ok'] else 'VIOLATION'}")
    print("\n  Why this beats the earlier 40B/15T draft: at a 4T budget the REAL native")
    print("  supply covers a MAJORITY of the Indic lane. Inflating the budget to 15T is")
    print("  what forced that draft to 60% non-native. Smaller budget = more honest Indic.")

    # ---------------------------------------------------------------- 4
    hdr("4 · CURRICULUM — STAGES, CONTEXT, AND STAGE x LANE RECONCILIATION")
    stg = CU.stage_tokens()
    print(f"{'stage':<18}{'frac':>7}{'tokens':>9}{'window':>10}{'span':>18}  note")
    for s in stg:
        print(f"{s['name']:<18}{s['frac']*100:>6.2f}%{T(s['tokens']):>9}"
              f"{s['ctx']:>9,}{T(s['start'])+'->'+T(s['end']):>18}  {s['note']}")

    print("\n  Stage x lane integration (does the schedule reproduce the global mixture?)")
    integ = CU.integrate()
    print(f"  {'lane':<20}{'S0':>7}{'S1':>7}{'S2':>7}{'S3':>7}{'S4':>7}"
          f"{'integr.':>9}{'target':>8}{'drift':>8}")
    for lane in C.MIXTURE:
        cells = "".join(f"{CU.STAGE_MIX[s['name']].get(lane,0)*100:>6.1f}%" for s in stg)
        v = integ[lane]
        print(f"  {lane:<20}{cells}{v['share']*100:>8.2f}%{v['target_share']*100:>7.1f}%"
              f"{v['delta']*100:>+7.2f}pp")

    print("\n  Gradient-safe blending bands (V4 saw a ~150x grad-norm spike on an")
    print("  abrupt Hindi share change -- every transition is ramped, not stepped):")
    print(f"  Rate limit: {CU.MAX_PP_PER_100B:.0f}pp per 100B tokens.")
    for bd in CU.blend_bands():
        flag = "OK" if bd["within_limit"] else (
            "steep, but LR->0 here" if bd["lr_protected"] else "*** EXCEEDS LIMIT ***")
        print(f"    {bd['frm']:<16} -> {bd['to']:<16} worst '{bd['worst_lane']}' "
              f"{bd['worst_delta']*100:+5.1f}pp | ramp {G(bd['ramp_tokens']):>7} straddling "
              f"({G(bd['tail_of_prev'])} tail + {G(bd['head_of_next'])} head "
              f"= {bd['pct_of_next']*100:.0f}% of next) | {bd['pp_per_100B']:>4.1f}pp/100B [{flag}]")

    print("\n  Long-context pool (selected from EXISTING lanes -- not a new token slot):")
    for d in CU.LONG_CTX_POOL:
        print(f"    {d['lane']:<20}{d['doc']:<45}{d['why']}")

    # ---------------------------------------------------------------- 5
    hdr("5 · DIFFICULTY AND REASONING-LENGTH BANDS")
    for bd in CU.BANDS:
        print(f"\n  Band {bd['band']} — {bd['name']}  ({bd['steps']} steps, "
              f"{bd['tokens']} reasoning tokens)")
        print(f"    example : {bd['example']}")
        print(f"    lanes   : {bd['lanes']}")
        print(f"    schedule: {bd['sched']}")
    print("\n  Effort dial (controllable reasoning depth objective):")
    for k, v in CU.EFFORT_LEVELS.items():
        mix = ", ".join(f"B{b}:{p*100:.0f}%" for b, p in v["band_mix"].items())
        print(f"    {k:<8}{v['budget']:<22}{mix}")

    # ---------------------------------------------------------------- 6
    hdr("6 · PROTECTED FLOOR AND ANNEAL RESERVE")
    fl = F.floor_report(rows)
    print(f"Floor: {fl['pct']*100:.0f}% of EVERY batch = "
          f"{fl['seqs_per_batch']} of {fl['batch_seqs']} sequences, drawn from "
          f"{fl['lanes']}")
    print("  Tightened definition vs V4: translated/synthetic do NOT count. Otherwise the")
    print("  selector satisfies the floor with cheap generated text and the scarce native")
    print("  tiers are never touched -- the exact erasure the floor exists to prevent.")
    print(f"  Per-stage native-Indic availability:")
    for s, v in fl["per_stage"].items():
        mark = "<-- binding" if s == fl["binding_stage"] else ""
        print(f"    {s:<18}{v*100:>6.1f}%  {mark}")
    print(f"  Satisfiable: {fl['satisfiable']}  (slack {fl['slack']*100:+.1f}pp)")

    rr = F.reserve_report()
    print(f"\nAnneal reserve: {T(rr['total'])} = {rr['pct_of_budget']*100:.1f}% of budget, "
          f"held BACK from the lane budgets")
    print(f"  {'component':<48}{'share':>7}{'tokens':>9}")
    for r in rr["rows"]:
        print(f"  {r['name']:<48}{r['share']*100:>6.0f}%{T(r['tokens']):>9}")
    print(f"  EXCLUDED from the reserve: {', '.join(rr['exclusions'])}")

    vn = F.verified_native_exposure(rows)
    print(f"\n  Verified-native exposure check (the bug that sinks most drafts):")
    print(f"    unique supply     {T(vn['unique'])}")
    print(f"    lane budget       {T(vn['bulk_tokens'])}  -> {vn['epochs']:.2f} epochs")
    print(f"    anneal component  {T(vn['anneal_tokens'])}  "
          f"= DEFERRED 2nd pass, carved OUT of the lane budget (not added)")
    print(f"    cap {vn['cap']:.1f}x -> {'PASS' if vn['ok'] else 'FAIL'}")

    # ---------------------------------------------------------------- 7
    hdr("7 · FERTILITY -> WHAT THE INDIC BUDGET ACTUALLY BUYS")
    print(f"Vocabulary: {C.VOCAB_SIZE:,} (3 x 2^16), frozen pre-training")
    print(f"Weighted Indic fertility: {F.weighted_indic_fertility():.4f} tokens/word "
          f"(target <= 1.75)")
    ew = F.effective_words(ind["total"])
    print(f"\n  Same {T(ind['total'])} Indic token budget, in WORDS learned:")
    print(f"  {'tokenizer':<26}{'fertility':>10}{'words':>10}{'vs en-BPE':>11}")
    for k, v in ew.items():
        print(f"  {k:<26}{v['fertility']:>10.2f}{T(v['words']):>10}"
              f"{v['vs_english_bpe']:>10.2f}x")
    s2 = F.session2_balance_score()
    print(f"\n  Session-2 balance metric  Score = 1000/(Xmax-Xmin):")
    print(f"    Xmax {s2['x_max']:.2f} ({max(C.FERTILITY_TARGETS, key=C.FERTILITY_TARGETS.get)})  "
          f"Xmin {s2['x_min']:.2f} ({min(C.FERTILITY_TARGETS, key=C.FERTILITY_TARGETS.get)})  "
          f"spread {s2['spread']:.2f}  -> score {s2['score']:.1f}")

    # ---------------------------------------------------------------- 8
    hdr("8 · PROXY EXPERIMENTS — THE PLAN AS A TESTABLE HYPOTHESIS")
    for p in P.proxy_table():
        print(f"\n  [{p['id']}] {p['title']}  ({p['scale']}, {T(p['tokens'])}/arm, "
              f"{p['arms']} arms, ~{p['gpu_hours']:,.0f} GPU-h, ~${p['usd']:,.0f})")
        print(f"      H : {p['hypothesis']}")
        print(f"      A : {p['arm_a']}")
        print(f"      B : {p['arm_b']}")
        if p.get("arm_c"):
            print(f"      C : {p['arm_c']}")
        print(f"      M : {p['metric']}")
        print(f"      -> {p['rule']}")
    pt = P.proxy_total()
    print(f"\n  Total proxy cost ~${pt['total_usd']:,.0f} "
          f"({pt['total_gpu_hours']:,.0f} GPU-h) = "
          f"{pt['pct_of_flagship']*100:.2f}% of the flagship run (~${pt['flagship_usd']:,.0f}).")
    print("  Every ratio in this plan is a hypothesis until one of these fires.")

    # ---------------------------------------------------------------- 9
    hdr("9 · SENSITIVITY — WHAT BREAKS FIRST IF THE INVENTORY IS WRONG")
    print("  Targets are fixed; what moves is how much of each is REAL. Gap = must generate.")
    print(f"\n{'scenario':<42}{'vn ep':>7}{'real nat':>9}{'nat%':>7}"
          f"{'Indic gap':>11}{'real%':>7}{'floor':>7}")
    print("-" * 78)
    for sc in S.sensitivity_table():
        print(f"{sc['name']:<42}{sc['vn_epochs']:>6.2f}x{T(sc['real_native']):>9}"
              f"{sc['real_native_pct']*100:>6.1f}%{T(sc['indic_gap']):>11}"
              f"{sc['total_real_pct']*100:>6.1f}%{'OK' if sc['floor_ok'] else 'FAIL':>7}")
    print("\n  Stated responses (the plan bends; it does not break):")
    for lane, txt in S.RESPONSES.items():
        print(f"\n    IF {lane} supply misses:")
        for line in [txt[i:i+72] for i in range(0, len(txt), 72)]:
            print(f"      {line}")

    hdr("10 · CLEANING PRIORITY — AIMED AT THE STARVED SLOTS")
    cp = S.cleaning_plan(rows)
    print(f"{'lane':<20}{'yield':>7}{'clean need':>12}{'collect tgt':>13}"
          f"{'RAW required':>14}{'starved':>9}")
    print("-" * 78)
    for d in cp:
        print(f"{d['lane']:<20}{d['yield_rate']*100:>6.0f}%{T(d['clean_needed']):>12}"
              f"{T(d['collected_target']):>13}{T(d['raw_required']):>14}"
              f"{'YES' if d['starved'] else '':>9}")
    print("-" * 78)
    print(f"{'TOTAL RAW INTAKE':<20}{'':>7}{T(sum(d['clean_needed'] for d in cp)):>12}"
          f"{T(sum(d['collected_target'] for d in cp)):>13}"
          f"{T(sum(d['raw_required'] for d in cp)):>14}")
    print("\n  Marginal value of one more cleaned token (where to point the cohort):")
    for i, d in enumerate(S.priority_queue(rows)[:6], 1):
        tag = " [floor lane -- counts double]" if d["floor_lane"] else ""
        print(f"    {i}. {d['lane']:<20} score {d['score']:>6.2f}  "
              f"(supply headroom {d['headroom']:.2f}x, {d['epochs']:.2f} epochs){tag}")

    hdr("11 · VALIDATION — INTERNAL CONSISTENCY")
    c = V.run_all(rows)
    for name, ok, detail in c.rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")
    print(f"\n  {len(c.rows)-len(c.failed)}/{len(c.rows)} checks passed.")
    return 0 if not c.failed else 1


if __name__ == "__main__":
    sys.exit(main())
