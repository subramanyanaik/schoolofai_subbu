"""E6 -- swap a 50,257-token vocabulary for a 1,000,000-token one, mid-flight.

Delta parameters must be exactly zero.  A dense head cannot do this at all
without allocating 384M new parameters and retraining them from scratch.

Experiment-design trap: E6 must swap the input embedding *and* the head.
Swapping only the head leaves the input table at |V|=50,257 while remapped ids
run to 1M -- an out-of-bounds gather.  Here both pathways share one Occupancy,
so rebuilding it swaps both at once.

The bpb cost is real and is reported: probability mass now spreads over ~20x
more classes the model never trained on.  The claim is feasibility at zero
parameter cost, not a quality gain.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _common import CHECKPOINTS, rule, table, write_json
from kronecker_v2.data import DEFAULT_DATA_DIR, TokenStream, env_device, resolve_paths
from kronecker_v2.eval import evaluate
from kronecker_v2.model import GPT
from kronecker_v2.train import build_arm
from kronecker_v2.vocab import PackedVocab, build_large_vocab, gpt2_token_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fineweb")
    ap.add_argument("--arm", default="a2_kronf_kas")
    ap.add_argument("--target-vocab", type=int, default=1_000_000)
    ap.add_argument("--d-c", type=int, default=32)
    ap.add_argument("--L-max", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=100)
    ap.add_argument("--swap-batch-size", type=int, default=1,
                    help="eval batch for the 1M-vocab pass; the logits tensor "
                         "alone is |V| x N x 4 bytes, so this must be small")
    args = ap.parse_args()

    device = env_device()
    ckpt_path = CHECKPOINTS / f"{args.arm}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} missing -- run e4_train_arms.py first")

    base_toks = gpt2_token_bytes()
    train_path, val_path = resolve_paths(args.source, DEFAULT_DATA_DIR)

    rule("E6.1 -- the trained model at its original vocabulary")
    model, packed, codec = build_arm(
        args.arm, base_toks, d_c=args.d_c, L_max=args.L_max,
        block_size=args.block_size,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    params_before = model.n_params()
    blen = packed.true_lengths.astype(np.float64)
    val_stream = TokenStream(val_path, args.block_size, seed=0)
    # Same batch size and batch count as the swapped pass, so both bpb numbers
    # are measured over the identical window set.
    before = evaluate(model, val_stream, device, blen,
                      batch_size=args.swap_batch_size,
                      max_batches=args.eval_batches)
    print(f"  vocab {packed.V:,}   params {params_before:,}   bpb {before['bpb']:.4f}")
    # Free the base model before allocating the 1M-vocab one; on a small card
    # holding both at once is what actually runs out of memory.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------ grow the vocab
    rule(f"E6.2 -- growing the vocabulary to {args.target_vocab:,}")
    texts = []
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = np.memmap(train_path, dtype=np.uint32, mode="r")[:4_000_000]
    orig = packed.order[np.asarray(ids, dtype=np.int64)]
    texts.append(enc.decode(orig.tolist()))
    big_toks = build_large_vocab(
        args.target_vocab, base_toks, texts, L_max=args.L_max, seed=0
    )
    stats = build_large_vocab.last_stats
    print(f"  base {stats['base']:,} + harvested {stats['harvested']:,} "
          f"+ synthetic {stats['synthetic']:,} = {stats['total']:,}")
    big_packed = PackedVocab(big_toks, L_max=args.L_max)
    print(f"  occupancy: {big_packed.total_bytes:,} nonzeros "
          f"(~{big_packed.total_bytes * 12 / 1e6:.0f} MB)")

    # ------------------------------------------------- rebuild at 1M vocab
    rule("E6.3 -- the same weights, a 1M-token vocabulary")
    big_model = GPT(
        args.arm, big_packed.V, big_packed, codec=codec,
        block_size=args.block_size,
    )
    # Every trainable tensor is vocabulary-independent, so this is a plain copy.
    missing, unexpected = big_model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not [k for k in missing if "occ" not in k], missing
    big_model.to(device)
    params_after = big_model.n_params()
    delta = params_after - params_before
    print(f"  params before / after : {params_before:,} / {params_after:,}")
    print(f"  delta parameters      : {delta:+,}")

    # Remap the validation ids: base-packed -> original gpt2 -> large-packed.
    base_to_gpt2 = packed.order
    gpt2_to_big = big_packed.inv_order[: len(base_toks)]
    remap = gpt2_to_big[base_to_gpt2].astype(np.uint32)

    class RemappedStream(TokenStream):
        def __init__(self, inner, table):
            self.data = table[np.asarray(inner.data, dtype=np.int64)]
            self.block_size = inner.block_size
            self.rng = np.random.default_rng(0)
            self.path = inner.path

    big_stream = RemappedStream(TokenStream(val_path, args.block_size, seed=0), remap)
    after = evaluate(big_model, big_stream, device,
                     big_packed.true_lengths.astype(np.float64),
                     batch_size=args.swap_batch_size,
                     max_batches=args.eval_batches)
    print(f"  bpb before / after    : {before['bpb']:.4f} / {after['bpb']:.4f}")

    dense_needed = 384 * args.target_vocab
    rule("E6 summary")
    table(
        [
            ["vocabulary", f"{packed.V:,} -> {big_packed.V:,}"],
            ["params before / after", f"{params_before:,} / {params_after:,}"],
            ["delta parameters", f"{delta:+,}"],
            ["occupancy matrix", f"{big_packed.total_bytes:,} nonzeros"],
            ["a dense head would need", f"{dense_needed / 1e6:.0f}M params, "
                                        f"{dense_needed * 12 / 1e9:.1f} GB Adam state"],
            ["bpb, base -> swapped", f"{before['bpb']:.4f} -> {after['bpb']:.4f}"],
        ],
        ["", ""],
    )

    gates = {"delta_params_is_zero": delta == 0}
    print(f"\n  {'PASS' if gates['delta_params_is_zero'] else 'FAIL'}  delta_params == 0")

    write_json("e6_vocab_swap.json", {
        "arm": args.arm,
        "vocab_before": packed.V,
        "vocab_after": big_packed.V,
        "params_before": params_before,
        "params_after": params_after,
        "delta_params": delta,
        "occupancy_nnz": big_packed.total_bytes,
        "occupancy_mb": big_packed.total_bytes * 12 / 1e6,
        "dense_head_would_need_params": dense_needed,
        "dense_head_would_need_adam_gb": dense_needed * 12 / 1e9,
        "bpb_before": before["bpb"],
        "bpb_after": after["bpb"],
        "vocab_composition": stats,
        "gates": gates,
        "note": "bpb rises because mass now spreads over ~20x more classes the "
                "model never trained on; the claim is feasibility at zero "
                "parameter cost, not a quality gain",
    })
    if not gates["delta_params_is_zero"]:
        raise SystemExit("E6 gate failed")


if __name__ == "__main__":
    main()
