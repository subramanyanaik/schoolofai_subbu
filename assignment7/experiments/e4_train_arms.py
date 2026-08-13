"""E4 -- five arms, token-matched.

    a0_bpe_tied        nn.Embedding            dense head, tied
    a1_v1_dense        V1 codec (published)    dense head
    a2_kronf_kas       Kronecker-F codec       KAS
    a3_kronf_kas_tied  Kronecker-F codec       KAS, W_out = W_proj
    a4_kas_unigram     Kronecker-F codec       KAS + one scalar per token

Trap 7: arms differ in throughput by ~2.8x on this hardware, so a per-arm
wall-clock cap would silently compare them at different token counts.  A
calibration probe measures throughput, one affordable budget is chosen, and it
is then held fixed for every arm.  Pin --tokens to match another machine.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _common import CHECKPOINTS, rule, table, write_json
from kronecker_v2.data import DEFAULT_DATA_DIR, TokenStream, env_device, resolve_paths
from kronecker_v2.eval import evaluate, unigram_floor
from kronecker_v2.model import ARMS
from kronecker_v2.train import build_arm, choose_token_budget, set_seed, throughput_probe, train
from kronecker_v2.vocab import gpt2_token_bytes

EVAL_BATCHES = 200


def output_row_norm_stats(model) -> dict | None:
    """The E4 diagnosis, measured rather than asserted.

    A dense head encodes each token's unigram log-prior for free in the
    *magnitude* of its row.  The codec's 1/sqrt(L) normalisation makes every
    KAS output row exactly unit norm, so KAS has no per-token magnitude at all.
    """
    occ = getattr(model.lm_head, "occ", None)
    if occ is None:
        return None
    vals = occ.vals.detach().cpu().numpy().astype(np.float64)
    offs = occ.offsets.detach().cpu().numpy()
    norms = np.sqrt(np.add.reduceat(vals**2, offs))
    return {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "std": float(norms.std()),
        "note": "every output row is unit norm => no per-token frequency prior",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fineweb")
    ap.add_argument("--tokens", type=int, default=None,
                    help="pin the per-arm token budget (needed to match another run)")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="minutes per arm; the slowest arm sets the budget for all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--d-c", type=int, default=32)
    ap.add_argument("--L-max", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="e4_train_arms.json")
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint filenames, so a second "
                         "configuration does not clobber the checkpoints "
                         "E6/E7/E8 were measured from")
    args = ap.parse_args()

    if args.tokens is None and args.budget_min is None:
        args.budget_min = 5.0

    device = env_device()
    arms = args.arms.split(",")
    toks = gpt2_token_bytes()
    train_path, val_path = resolve_paths(args.source, DEFAULT_DATA_DIR)
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} missing -- run experiments/prepare_data.py --source {args.source}"
        )

    rule(f"E4 -- five arms, token-matched.  device={device}, corpus={args.source}")
    common = dict(
        d_c=args.d_c, L_max=args.L_max, d_model=args.d_model,
        n_layer=args.n_layer, n_head=args.n_head, block_size=args.block_size,
    )

    # ------------------------------------------------- calibration probe
    tokens = args.tokens
    rates: dict[str, float] = {}
    if tokens is None:
        print("calibration probe (trap 7: choose one budget every arm can afford)")
        for arm in arms:
            set_seed(args.seed)
            model, _, _ = build_arm(arm, toks, seed=args.seed, **common)
            model.to(device)
            stream = TokenStream(train_path, args.block_size, seed=args.seed)
            rates[arm] = throughput_probe(model, stream, device, args.batch_size)
            print(f"  {arm:<20s} {rates[arm]:>10,.0f} tok/s")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        tokens = choose_token_budget(
            rates, args.budget_min, args.batch_size * args.block_size
        )
    print(f"\nper-arm token budget: {tokens:,} (identical for every arm)")

    # ----------------------------------------------------------- training
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for arm in arms:
        rule(f"training {arm}")
        set_seed(args.seed)
        model, packed, codec = build_arm(arm, toks, seed=args.seed, **common)
        model.to(device)
        train_stream = TokenStream(train_path, args.block_size, seed=args.seed)
        val_stream = TokenStream(val_path, args.block_size, seed=args.seed)
        blen = packed.true_lengths.astype(np.float64)

        def eval_fn(m):
            return evaluate(m, val_stream, device, blen,
                            batch_size=args.batch_size, max_batches=40)

        total_steps = max(1, tokens // (args.batch_size * args.block_size))
        stats = train(
            model, train_stream, device, tokens,
            batch_size=args.batch_size, lr=args.lr,
            eval_fn=eval_fn, eval_every=max(1, total_steps // 4),
            log_every=max(1, total_steps // 10),
            on_log=lambda r: print(
                "  step {step:>6}  loss {train_loss:.4f}".format(**r)
                + (f"  val bpb {r['val_bpb']:.4f}" if "val_bpb" in r else "")
            ),
        )
        final = evaluate(model, val_stream, device, blen,
                         batch_size=args.batch_size, max_batches=EVAL_BATCHES)
        bd = model.param_breakdown()
        rec = {
            "arm": arm,
            "tokens": stats["tokens"],
            "steps": stats["steps"],
            "wall_s": stats["wall_s"],
            "tokens_per_s": stats["tokens_per_s"],
            "final_train_loss": stats["final_train_loss"],
            "val": final,
            "params": bd,
            "codec": repr(codec) if codec else None,
            "unused_params": model.unused_parameters(),
            "output_row_norms": output_row_norm_stats(model),
            "history": stats["history"],
        }
        results[arm] = rec
        print(f"\n  val bpb {final['bpb']:.4f}  ppl {final['ppl']:.1f}  "
              f"params {bd['total'] / 1e6:.2f}M  {stats['tokens_per_s']:,.0f} tok/s")

        torch.save(
            {"arm": arm, "state_dict": model.state_dict(), "config": {**common,
             "vocab_size": packed.V, "seed": args.seed}},
            CHECKPOINTS / f"{arm}{args.tag}.pt",
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------- the floor
    rule("unigram-only floor")
    packed0 = build_arm("a0_bpe_tied", toks, seed=args.seed, **common)[1]
    train_ids = np.memmap(train_path, dtype=np.uint32, mode="r")[: 8_000_000]
    val_stream = TokenStream(val_path, args.block_size, seed=args.seed)
    floor = unigram_floor(
        train_ids, val_stream, packed0.true_lengths.astype(np.float64),
        packed0.V, batch_size=args.batch_size, max_batches=EVAL_BATCHES,
    )
    print(f"  unigram floor: {floor['bpb']:.4f} bpb "
          f"({floor['nll_nats_per_token']:.4f} nats/token)")

    # ---------------------------------------------------------- summary
    rule("E4 summary")
    rows = []
    for arm in arms:
        r = results[arm]
        rows.append([
            arm, f"{r['val']['bpb']:.4f}", f"{r['val']['ppl']:.1f}",
            f"{r['params']['total'] / 1e6:.2f}M",
            f"{r['params']['head'] / 1e6:.2f}M",
            f"{r['tokens_per_s']:,.0f}",
        ])
    rows.append(["unigram-only floor", f"{floor['bpb']:.4f}", "-", "-", "-", "-"])
    table(rows, ["arm", "val bpb", "ppl", "params", "head params", "tok/s"])

    payload = {
        "config": vars(args),
        "device": device,
        "tokens_per_arm": tokens,
        "probe_rates": rates,
        "arms": results,
        "unigram_floor": floor,
    }
    if "a0_bpe_tied" in results and "a2_kronf_kas" in results:
        a0, a2 = results["a0_bpe_tied"]["val"]["bpb"], results["a2_kronf_kas"]["val"]["bpb"]
        payload["a2_minus_a0_bpb"] = a2 - a0
        payload["a2_vs_unigram_floor_bpb"] = a2 - floor["bpb"]
        if "a4_kas_unigram" in results:
            a4 = results["a4_kas_unigram"]["val"]["bpb"]
            payload["a4_minus_a0_bpb"] = a4 - a0
            payload["gap_recovered_by_unigram_bias"] = (a2 - a4) / (a2 - a0) if a2 != a0 else None
    write_json(args.out, payload)


if __name__ == "__main__":
    main()
