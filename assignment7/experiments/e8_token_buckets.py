"""E8 -- where the fixed surface-form subspace helps, and where it hurts.

Delta = NLL(KAS) - NLL(dense).  Negative means KAS wins.

The design-time prediction was that frequent tokens need the unigram prior KAS
cannot express, while rare tokens need the surface-form generalisation the
dense head never learns -- so the sign should flip somewhere in the tail.  This
measures it, by validation-set frequency decile.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _common import CHECKPOINTS, rule, table, write_json
from kronecker_v2.data import DEFAULT_DATA_DIR, TokenStream, env_device, resolve_paths
from kronecker_v2.eval import per_token_nll
from kronecker_v2.train import build_arm
from kronecker_v2.vocab import gpt2_token_bytes

N_DECILES = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fineweb")
    ap.add_argument("--baseline", default="a0_bpe_tied")
    ap.add_argument("--arms", default="a2_kronf_kas,a3_kronf_kas_tied,a4_kas_unigram")
    ap.add_argument("--d-c", type=int, default=32)
    ap.add_argument("--L-max", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=200)
    args = ap.parse_args()

    device = env_device()
    toks = gpt2_token_bytes()
    _, val_path = resolve_paths(args.source, DEFAULT_DATA_DIR)

    def measure(arm):
        model, packed, _ = build_arm(
            arm, toks, d_c=args.d_c, L_max=args.L_max, block_size=args.block_size
        )
        ckpt = torch.load(CHECKPOINTS / f"{arm}.pt", map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        stream = TokenStream(val_path, args.block_size, seed=0)
        nll, counts = per_token_nll(model, stream, device, packed.V,
                                    max_batches=args.eval_batches)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return nll, counts, packed

    rule(f"E8 -- baseline {args.baseline}")
    base_nll, counts, packed = measure(args.baseline)

    # Deciles by validation frequency, over tokens that actually occur.
    seen = np.flatnonzero(counts > 0)
    order = seen[np.argsort(-counts[seen], kind="stable")]
    cum = np.cumsum(counts[order])
    edges = np.searchsorted(cum, np.linspace(0, cum[-1], N_DECILES + 1)[1:])
    buckets = np.split(order, edges[:-1])
    print(f"  {seen.size:,} distinct tokens observed, "
          f"{int(counts.sum()):,} occurrences, {N_DECILES} equal-mass deciles")

    payload = {
        "baseline": args.baseline,
        "n_distinct_tokens": int(seen.size),
        "n_occurrences": int(counts.sum()),
        "deciles": [
            {"decile": i, "n_types": int(b.size),
             "n_occurrences": int(counts[b].sum()),
             "mean_freq": float(counts[b].mean()),
             "mean_byte_len": float(packed.true_lengths[b].mean())}
            for i, b in enumerate(buckets)
        ],
        "arms": {},
    }

    header = ["arm"] + [f"d{i}" for i in range(N_DECILES)]
    rows = []
    for arm in args.arms.split(","):
        if not (CHECKPOINTS / f"{arm}.pt").exists():
            print(f"skip {arm}: no checkpoint")
            continue
        rule(f"E8 -- {arm}")
        arm_nll, arm_counts, _ = measure(arm)
        assert np.array_equal(counts, arm_counts), "arms must see identical windows"
        deltas = []
        for b in buckets:
            n = counts[b].sum()
            deltas.append(float((arm_nll[b].sum() - base_nll[b].sum()) / max(n, 1)))
        payload["arms"][arm] = {
            "delta_nll_by_decile": deltas,
            "overall_delta_nll": float(
                (arm_nll[seen].sum() - base_nll[seen].sum()) / counts.sum()
            ),
            "sign_flips": bool(min(deltas) < 0 < max(deltas)),
            "best_decile": int(np.argmin(deltas)),
            "worst_decile": int(np.argmax(deltas)),
        }
        rows.append([arm] + [f"{d:+.3f}" for d in deltas])

    rule("E8 summary -- delta NLL vs the dense head (negative = KAS wins)")
    print("decile 0 = most frequent tokens, decile 9 = rarest\n")
    table(rows, header)
    for arm, rec in payload["arms"].items():
        print(f"\n{arm}: overall {rec['overall_delta_nll']:+.4f} nats/token, "
              f"best decile {rec['best_decile']}, worst decile {rec['worst_decile']}, "
              f"sign flips: {rec['sign_flips']}")

    write_json("e8_token_buckets.json", payload)


if __name__ == "__main__":
    main()
