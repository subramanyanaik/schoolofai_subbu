"""E5 -- capacity.  Which knob actually predicts language-model quality?

The design-time claim was "D is the only capacity knob".  This sweep tests it
by varying d_c and L_max independently at overlapping D.

Confound, stated rather than glossed: D = d_c * L_max by construction, so
neither factor can be varied in isolation.  Moving d_c also moves codebook
coherence; moving L_max also moves the exact-length limit.  Both axes are
reported for exactly that reason.

Trap 8: this script MERGES into its results file, keyed by (d_c, L_max).  A
timed-out sweep once rewrote its own output with a single partial run and the
next invocation merged that partial forward -- five completed runs gone.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _common import merge_json, read_json, rule, table, write_json
from kronecker_v2.codebooks import byte_codebook, coherence_stats
from kronecker_v2.data import DEFAULT_DATA_DIR, TokenStream, env_device, resolve_paths
from kronecker_v2.eval import evaluate
from kronecker_v2.train import build_arm, set_seed, train
from kronecker_v2.vocab import gpt2_token_bytes

GRID = [
    (128, 32),   # D = 4096
    (64, 32),    # D = 2048
    (64, 128),   # D = 8192
    (32, 32),    # D = 1024
    (32, 64),    # D = 2048
    (32, 128),   # D = 4096  <- the configuration E4 used
    (16, 128),   # D = 2048
]
OUT = "e5_d_sweep.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fineweb")
    ap.add_argument("--tokens", type=int, default=1_500_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="ignore cached runs")
    args = ap.parse_args()

    device = env_device()
    toks = gpt2_token_bytes()
    train_path, val_path = resolve_paths(args.source, DEFAULT_DATA_DIR)
    existing = (read_json(OUT) or {}).get("runs", {}) if not args.redo else {}

    rule(f"E5 -- d_c / L_max sweep, {args.tokens:,} tokens per run (reduced budget)")
    print("comparable within this sweep only -- not against E4.\n")

    updates = {}
    for d_c, L_max in GRID:
        key = f"d_c={d_c},L_max={L_max}"
        if key in existing:
            print(f"skip {key} (already in results)")
            continue
        coh = coherence_stats(byte_codebook(d_c, seed=args.seed))
        print(f"\n--- {key}  D={d_c * L_max}  max coherence {coh['max_coherence']:.4f}")
        set_seed(args.seed)
        model, packed, codec = build_arm(
            "a2_kronf_kas", toks, d_c=d_c, L_max=L_max, seed=args.seed,
            block_size=args.block_size,
        )
        model.to(device)
        train_stream = TokenStream(train_path, args.block_size, seed=args.seed)
        val_stream = TokenStream(val_path, args.block_size, seed=args.seed)
        blen = packed.true_lengths.astype(np.float64)
        steps = max(1, args.tokens // (args.batch_size * args.block_size))
        stats = train(
            model, train_stream, device, args.tokens,
            batch_size=args.batch_size, log_every=max(1, steps // 5),
            on_log=lambda r: print("    step {step:>6}  loss {train_loss:.4f}".format(**r)),
        )
        final = evaluate(model, val_stream, device, blen,
                         batch_size=args.batch_size, max_batches=200)
        updates[key] = {
            "d_c": d_c, "L_max": L_max, "D": codec.D,
            "bpb": final["bpb"], "ppl": final["ppl"],
            "params": model.param_breakdown()["total"],
            "head_params": model.param_breakdown()["head"],
            "max_coherence": coh["max_coherence"],
            "mean_coherence": coh["mean_coherence"],
            "welch_bound": coh["welch_bound"],
            "n_truncated": packed.n_truncated,
            "tokens": stats["tokens"],
            "tokens_per_s": stats["tokens_per_s"],
        }
        print(f"    -> bpb {final['bpb']:.4f}  params {model.param_breakdown()['total'] / 1e6:.2f}M")
        merge_json(OUT, updates)  # write after every run, not at the end
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = read_json(OUT) or {}
    runs = payload.get("runs", {})
    ordered = sorted(runs.values(), key=lambda r: r["bpb"])

    rule("E5 -- sorted by quality")
    table(
        [[r["d_c"], r["L_max"], r["D"], f"{r['bpb']:.4f}",
          f"{r['params'] / 1e6:.2f}M", f"{r['max_coherence']:.3f}"] for r in ordered],
        ["d_c", "L_max", "D", "bpb", "params", "max coherence"],
    )

    coh_vals = [r["max_coherence"] for r in ordered]
    D_vals = [r["D"] for r in ordered]
    bpb_vals = [r["bpb"] for r in ordered]
    coh_monotone = all(a <= b for a, b in zip(coh_vals, coh_vals[1:]))

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    analysis = {
        "coherence_orders_bpb_monotonically": bool(coh_monotone),
        "spearman_coherence_vs_bpb": spearman(coh_vals, bpb_vals),
        "spearman_D_vs_bpb": spearman(D_vals, bpb_vals),
        "best": ordered[0] if ordered else None,
        "e4_config_rank": next(
            (i + 1 for i, r in enumerate(ordered)
             if r["d_c"] == 32 and r["L_max"] == 128), None),
        "n_runs": len(ordered),
    }
    print(f"\nmax coherence orders bpb monotonically : {coh_monotone}")
    print(f"spearman(coherence, bpb) = {analysis['spearman_coherence_vs_bpb']:+.3f}")
    print(f"spearman(D,          bpb) = {analysis['spearman_D_vs_bpb']:+.3f}")
    if analysis["e4_config_rank"]:
        print(f"\nthe configuration E4 used (d_c=32, L_max=128) ranks "
              f"{analysis['e4_config_rank']} of {len(ordered)}")

    # Same-D slices are the cleanest evidence D is the wrong quantity.
    by_D: dict[int, list] = {}
    for r in runs.values():
        by_D.setdefault(r["D"], []).append(r)
    analysis["spread_at_equal_D"] = {
        str(D): {"n": len(v), "bpb_min": min(x["bpb"] for x in v),
                 "bpb_max": max(x["bpb"] for x in v),
                 "spread": max(x["bpb"] for x in v) - min(x["bpb"] for x in v)}
        for D, v in by_D.items() if len(v) > 1
    }
    payload["analysis"] = analysis
    payload["config"] = vars(args)
    write_json(OUT, payload)  # payload already carries every merged run


if __name__ == "__main__":
    main()
