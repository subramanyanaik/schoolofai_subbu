"""E7 -- out-of-domain, multi-script.  Where the surface-form head wins.

A dense output head has learned essentially nothing about Devanagari or Telugu
byte sequences.  The codec head sees the same bytes it always did.  The same
fixed subspace that costs quality in-domain buys generalisation out-of-domain:
the two results are the same mechanism with opposite signs.

CONFOUND, stated rather than glossed.  The KAS arms are substantially
higher-entropy models, and a flatter distribution is automatically better on
out-of-distribution text.  Some unknown fraction of any Hindi/Telugu margin is
calibration, not byte-level transfer, and this experiment cannot separate the
two.  The clean version holds predictive entropy fixed (temperature-matching
each arm on a held-out in-domain set first) and is not run here.  Treat the
magnitude as an upper bound on the true effect.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _common import CHECKPOINTS, ROOT, rule, table, write_json
from kronecker_v2.data import DEFAULT_DATA_DIR, TokenStream, env_device, resolve_paths
from kronecker_v2.eval import bpb_of_text, evaluate
from kronecker_v2.model import ARMS
from kronecker_v2.train import build_arm
from kronecker_v2.vocab import gpt2_token_bytes

OOD_DIR = ROOT / "data" / "ood"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fineweb")
    ap.add_argument("--d-c", type=int, default=32)
    ap.add_argument("--L-max", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=120_000)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    meta_path = OOD_DIR / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            "out-of-domain corpus missing -- run experiments/build_ood_corpus.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    device = env_device()
    toks = gpt2_token_bytes()
    _, val_path = resolve_paths(args.source, DEFAULT_DATA_DIR)

    rule("E7 -- bits per byte on out-of-domain multi-script text")
    langs = list(meta.keys())
    results, rows = {}, []
    for arm in args.arms.split(","):
        ckpt_path = CHECKPOINTS / f"{arm}.pt"
        if not ckpt_path.exists():
            print(f"skip {arm}: no checkpoint")
            continue
        model, packed, _ = build_arm(
            arm, toks, d_c=args.d_c, L_max=args.L_max, block_size=args.block_size
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        # In-domain reference, so the OOD numbers can be read against something.
        val_stream = TokenStream(val_path, args.block_size, seed=0)
        indom = evaluate(model, val_stream, device,
                         packed.true_lengths.astype(np.float64),
                         batch_size=8, max_batches=100)
        rec = {"in_domain_bpb": indom["bpb"], "in_domain_ppl": indom["ppl"]}
        for lang in langs:
            text = (ROOT / meta[lang]["path"]).read_text(encoding="utf-8")
            r = bpb_of_text(model, text, packed, device,
                            block_size=args.block_size, max_tokens=args.max_tokens)
            rec[lang] = r["bpb"]
        results[arm] = rec
        rows.append([arm, f"{indom['ppl']:>8.1f}"] + [f"{rec[l]:.4f}" for l in langs])
        print(f"  {arm:<20s} " + "  ".join(f"{l} {rec[l]:.4f}" for l in langs))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    rule("E7 summary -- bits per byte (lower is better)")
    table(rows, ["arm", "in-dom ppl"] + [f"{l} ({meta[l]['label']})" for l in langs])

    deltas = {}
    if "a0_bpe_tied" in results:
        base = results["a0_bpe_tied"]
        rule("relative to the dense-head baseline (a0)")
        drows = []
        for arm, rec in results.items():
            if arm == "a0_bpe_tied":
                continue
            d = {l: 100 * (rec[l] - base[l]) / base[l] for l in langs}
            deltas[arm] = d
            drows.append([arm] + [f"{d[l]:+.1f}%" for l in langs])
        table(drows, ["arm"] + langs)

    write_json("e7_ood_multiscript.json", {
        "corpus": meta,
        "arms": results,
        "pct_vs_a0": deltas,
        "confound": (
            "the KAS arms are higher-entropy models and a flatter distribution "
            "is automatically better on out-of-distribution text; this design "
            "cannot separate byte-level transfer from calibration. Treat the "
            "margin as an upper bound. The clean version temperature-matches "
            "each arm on held-out in-domain data first and is not run here."
        ),
    })


if __name__ == "__main__":
    main()
