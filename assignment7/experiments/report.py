"""Print every number the README quotes, read straight from results/.

The README claims "every number below is read from a file in results/".  This
script is how that claim is checked: run it and diff against the write-up.
"""

from __future__ import annotations

from _common import read_json, rule, table


def e1() -> None:
    d = read_json("e1_invertibility.json")
    if not d:
        return
    rule("E1 -- invertibility")
    table(
        [[k, v["D"], f"{v['exact']:,} ({100 * v['exact_frac']:.3f}%)",
          f"{100 * v['exact_frac_within_L_max']:.3f}%",
          f"{v['n'] - v['within_L_max']:,}"]
         for k, v in d["gpt2_vocab"].items()],
        ["codec", "D", "exact / 50,257", "within L_max", "truncated"],
    )
    print(f"\nmean token length {d['vocab_stats']['mean_len']:.2f} bytes")
    table([[L, f"{v['truncate']:.3f}", f"{v['alias']:.3f}"]
           for L, v in d["overflow"].items()],
          ["token length", "truncate", "alias"])
    print(f"\nall gates pass: {d['all_gates_pass']}")


def e3() -> None:
    d = read_json("e3_cost.json")
    if not d:
        return
    rule("E3 -- cost")
    p = d["params"]
    vs = sorted(int(k) for k in p)
    table(
        [[f"{v:,}", f"{p[str(v)]['dense_head_params'] / 1e6:.1f}M",
          f"{p[str(v)]['kas_head_params'] / 1e6:.2f}M",
          f"{p[str(v)]['param_ratio']:.1f}x",
          f"{p[str(v)]['dense_adam_gb']:.2f} GB",
          f"{p[str(v)]['analytic_op_ratio']:.0f}x",
          f"{p[str(v)]['analytic_op_ratio_excl_projection']:.0f}x"] for v in vs],
        ["vocab", "dense head", "KAS head", "ratio", "dense Adam",
         "ops (strict)", "ops (excl proj)"],
    )
    t = d.get("timing", {})
    rows = [[v["device"], f"{v['V']:,}", v["N"],
             f"{v['dense_ms']:.2f}" if v.get("dense_ms") else "OOM",
             f"{v['kas_ms']:.2f}" if v.get("kas_ms") else "OOM",
             f"{v['speedup']:.2f}x" if v.get("speedup") else "-"]
            for v in t.values()]
    table(rows, ["device", "vocab", "N", "dense ms", "KAS ms", "KAS speedup"])
    if "loop_trap" in d:
        lt = d["loop_trap"]
        print(f"\ntrap 4: slice-add loop {lt['loop_ms']:.2f} ms vs "
              f"sparse-matmul {lt['bag_ms']:.2f} ms "
              f"({lt['loop_to_bag_speedup']:.1f}x) vs dense {lt['dense_ms']:.2f} ms")


def e4() -> None:
    d = read_json("e4_train_arms.json")
    if not d:
        return
    rule(f"E4 -- {d['tokens_per_arm']:,} tokens per arm")
    rows = [[a, f"{r['val']['bpb']:.4f}", f"{r['val']['ppl']:.1f}",
             f"{r['params']['total'] / 1e6:.2f}M",
             f"{r['params']['head'] / 1e6:.2f}M",
             f"{r['tokens_per_s']:,.0f}"] for a, r in d["arms"].items()]
    rows.append(["unigram-only floor", f"{d['unigram_floor']['bpb']:.4f}",
                 "-", "-", "-", "-"])
    table(rows, ["arm", "val bpb", "ppl", "params", "head params", "tok/s"])
    for k in ("a2_minus_a0_bpb", "a4_minus_a0_bpb", "a2_vs_unigram_floor_bpb",
              "gap_recovered_by_unigram_bias"):
        if d.get(k) is not None:
            print(f"  {k:<34s} {d[k]:+.4f}")
    rn = d["arms"].get("a2_kronf_kas", {}).get("output_row_norms")
    if rn:
        print(f"\noutput row norms: min {rn['min']:.6f}, max {rn['max']:.6f}, "
              f"std {rn['std']:.2e}")


def e4b() -> None:
    new = read_json("e4_best_config.json")
    old = read_json("e4_train_arms.json")
    if not new or not old:
        return
    rule("E4b -- the same KAS arms at the configuration E5 recommends")
    floor = old["unigram_floor"]["bpb"]
    a0 = old["arms"]["a0_bpe_tied"]["val"]["bpb"]
    rows = []
    for arm, rec in new["arms"].items():
        o = old["arms"].get(arm)
        nb = rec["val"]["bpb"]
        rows.append([
            arm,
            f"{o['val']['bpb']:.4f}" if o else "-",
            f"{nb:.4f}",
            f"{nb - o['val']['bpb']:+.4f}" if o else "-",
            f"{o['params']['total'] / 1e6:.2f}M" if o else "-",
            f"{rec['params']['total'] / 1e6:.2f}M",
            f"{nb - floor:+.4f}",
        ])
    table(rows, ["arm", "d_c=32,L=128", "d_c=64,L=32", "delta", "params before",
                 "params after", "vs floor"])
    print(f"\nbaseline A0 {a0:.4f} | unigram floor {floor:.4f}")
    print("A0 and A1 are not re-run: unaffected by d_c/L_max by construction.")


def e5() -> None:
    d = read_json("e5_d_sweep.json")
    if not d or not d.get("runs"):
        return
    rule("E5 -- capacity sweep, sorted by quality")
    ordered = sorted(d["runs"].values(), key=lambda r: r["bpb"])
    table([[r["d_c"], r["L_max"], r["D"], f"{r['bpb']:.4f}",
            f"{r['params'] / 1e6:.2f}M", f"{r['max_coherence']:.3f}",
            r["n_truncated"]] for r in ordered],
          ["d_c", "L_max", "D", "bpb", "params", "max coherence", "truncated"])
    a = d.get("analysis", {})
    print(f"\ncoherence orders bpb monotonically : {a.get('coherence_orders_bpb_monotonically')}")
    print(f"spearman(coherence, bpb)            : {a.get('spearman_coherence_vs_bpb'):+.3f}")
    print(f"spearman(D,          bpb)            : {a.get('spearman_D_vs_bpb'):+.3f}")
    for D, v in (a.get("spread_at_equal_D") or {}).items():
        print(f"  at D={D:>5}: {v['n']} configs span {v['spread']:.4f} bpb")


def e6() -> None:
    d = read_json("e6_vocab_swap.json")
    if not d:
        return
    rule("E6 -- vocabulary swap")
    table([
        ["vocabulary", f"{d['vocab_before']:,} -> {d['vocab_after']:,}"],
        ["params before / after", f"{d['params_before']:,} / {d['params_after']:,}"],
        ["delta parameters", f"{d['delta_params']:+,}"],
        ["occupancy matrix", f"{d['occupancy_nnz']:,} nonzeros "
                             f"(~{d['occupancy_mb']:.0f} MB)"],
        ["a dense head would need",
         f"{d['dense_head_would_need_params'] / 1e6:.0f}M params, "
         f"{d['dense_head_would_need_adam_gb']:.1f} GB Adam"],
        ["bpb, base -> swapped", f"{d['bpb_before']:.4f} -> {d['bpb_after']:.4f}"],
    ], ["", ""])


def e7() -> None:
    d = read_json("e7_ood_multiscript.json")
    if not d:
        return
    rule("E7 -- out-of-domain, multi-script (bits per byte)")
    langs = list(d["corpus"].keys())
    table([[a] + [f"{r[l]:.4f}" for l in langs] + [f"{r['in_domain_ppl']:.1f}"]
           for a, r in d["arms"].items()],
          ["arm"] + langs + ["in-dom ppl"])
    if d.get("pct_vs_a0"):
        table([[a] + [f"{v[l]:+.1f}%" for l in langs]
               for a, v in d["pct_vs_a0"].items()], ["vs a0"] + langs)


def e8() -> None:
    d = read_json("e8_token_buckets.json")
    if not d:
        return
    rule("E8 -- delta NLL by validation frequency decile (negative = KAS wins)")
    n = len(d["deciles"])
    table([[a] + [f"{x:+.3f}" for x in r["delta_nll_by_decile"]]
           for a, r in d["arms"].items()],
          ["arm"] + [f"d{i}" for i in range(n)])
    for a, r in d["arms"].items():
        print(f"  {a}: overall {r['overall_delta_nll']:+.4f}, "
              f"sign flips: {r['sign_flips']}")


def audit() -> None:
    d = read_json("params_audit.json")
    if not d:
        return
    rule("parameter audit")
    table([[a, f"{r['total'] / 1e6:.2f}M", f"{r['body'] / 1e6:.2f}M",
            f"{r['input_embedding'] / 1e6:.2f}M", f"{r['head'] / 1e6:.2f}M",
            f"{r['delta_params_8k_to_32k_vocab']:+,}",
            "OK" if not r["unused_params"] else "UNUSED"]
           for a, r in d["arms"].items()],
          ["arm", "total", "body", "input", "head", "delta V 8k->32k", "audit"])
    print(f"\nall gates pass: {d['all_gates_pass']}")


def main() -> None:
    for fn in (e1, e3, audit, e4, e4b, e5, e6, e7, e8):
        fn()


if __name__ == "__main__":
    main()
