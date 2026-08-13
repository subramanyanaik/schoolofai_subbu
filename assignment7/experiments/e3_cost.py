"""E3 -- cost.  Parameters, optimizer memory, op counts, and wall-clock.

The parameter and memory results are the defensible wins.  The wall-clock
result is mixed and is reported that way: a dense matmul is exactly what a GPU
is best at, and a 0.16%-dense sparse product does not beat it at high
arithmetic intensity.  The analytic op ratio is *not* a wall-clock claim.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from _common import rule, table, write_json
from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.kas_head import KASHead, analytic_op_counts
from kronecker_v2.vocab import PackedVocab, gpt2_token_bytes

D_MODEL = 768
D_C, L_MAX = 32, 128           # D = 4096
VOCABS = [50_257, 131_072, 262_144, 524_288, 1_000_000]
BYTES_PER_ADAM_SLOT = 4 * 3    # fp32 param + 2 fp32 moments


def synthetic_vocab(V: int, base: list[bytes], seed: int = 0) -> list[bytes]:
    """Grow to V surface forms, keeping the real GPT-2 length distribution.

    Length distribution is what drives B_V and therefore every op count here,
    so it is sampled from the real vocabulary rather than invented.
    """
    if V <= len(base):
        return base[:V]
    rng = np.random.default_rng(seed)
    lens = np.array([len(t) for t in base])
    extra = rng.choice(lens, size=V - len(base))
    flat = rng.integers(0, 256, size=int(extra.sum()), dtype=np.uint8).tobytes()
    out, pos = list(base), 0
    for L in extra:
        out.append(flat[pos : pos + int(L)])
        pos += int(L)
    return out


def bench(fn, warmup: int = 3, iters: int = 10, device: str = "cpu") -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main() -> None:
    payload: dict = {"d_model": D_MODEL, "d_c": D_C, "L_max": L_MAX, "D": D_C * L_MAX}
    base = gpt2_token_bytes()
    codec = KroneckerCodec(d_c=D_C, L_max=L_MAX)

    # --------------------------------------------------------- parameters
    rule("E3.1 -- output-head parameters and optimizer memory")
    rows, params = [], {}
    for V in VOCABS:
        packed = PackedVocab(synthetic_vocab(V, base), L_max=L_MAX)
        head = KASHead(D_MODEL, codec, packed, dtype=torch.float32)
        dense_p = D_MODEL * V
        kas_p = head.head_parameters()
        ops = analytic_op_counts(packed, D_MODEL, codec.D, N=1)
        rec = {
            "V": V,
            "dense_head_params": dense_p,
            "kas_head_params": kas_p,
            "param_ratio": dense_p / kas_p,
            "dense_adam_gb": dense_p * BYTES_PER_ADAM_SLOT / 1e9,
            "kas_adam_gb": kas_p * BYTES_PER_ADAM_SLOT / 1e9,
            "occupancy_nnz": packed.total_bytes,
            "occupancy_density": head.occ.density,
            "occupancy_mb": packed.total_bytes * 12 / 1e6,  # int64 col + fp32 val
            "analytic_op_ratio": ops["ratio"],
            "analytic_op_ratio_excl_projection": ops["ratio_excluding_projection"],
        }
        params[str(V)] = rec
        rows.append([
            f"{V:,}", f"{dense_p / 1e6:.1f}M", f"{kas_p / 1e6:.2f}M",
            f"{rec['param_ratio']:.1f}x", f"{rec['dense_adam_gb']:.2f} GB",
            f"{rec['analytic_op_ratio']:.0f}x",
            f"{rec['analytic_op_ratio_excl_projection']:.0f}x",
        ])
        del head
    table(rows, ["vocab", "dense head", "KAS head", "ratio", "dense Adam",
                 "op ratio", "op ratio (excl proj)"])
    payload["params"] = params

    p50 = params[str(50_257)]
    print(f"\noccupancy matrix at V=50,257: {p50['occupancy_nnz']:,} nonzeros, "
          f"{100 * p50['occupancy_density']:.3f}% dense, ~{p50['occupancy_mb']:.0f} MB")
    print(f"the dense head it replaces: {D_MODEL * 50257 * 4 / 1e6:.0f} MB of fp32 weights")

    # ---------------------------------------------------------- wall-clock
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    timing: dict = {}
    for device in devices:
        batch_sizes = [1, 256] if device == "cuda" else [64]
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        rule(f"E3.2 -- wall-clock, {device} ({dtype}).  ratio > 1 means KAS is faster")
        rows = []
        for V in VOCABS:
            packed = PackedVocab(synthetic_vocab(V, base), L_max=L_MAX)
            head = KASHead(D_MODEL, codec, packed, dtype=torch.float32).to(device)
            for N in batch_sizes:
                key = f"{device}_V{V}_N{N}"
                h = torch.randn(N, D_MODEL, device=device, dtype=torch.float32)
                entry = {"V": V, "N": N, "device": device}
                try:
                    W = torch.randn(V, D_MODEL, device=device, dtype=dtype)
                    entry["dense_ms"] = bench(
                        lambda: h.to(dtype) @ W.t(), device=device
                    )
                    del W
                except torch.cuda.OutOfMemoryError:
                    entry["dense_ms"] = None
                    entry["dense_note"] = "OOM"
                    torch.cuda.empty_cache()
                try:
                    if device == "cuda":
                        def run():
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                return head(h, impl="bag")
                    else:
                        def run():
                            return head(h, impl="bag")
                    entry["kas_ms"] = bench(run, device=device)
                except torch.cuda.OutOfMemoryError:
                    entry["kas_ms"] = None
                    entry["kas_note"] = "OOM"
                    torch.cuda.empty_cache()
                if entry["dense_ms"] and entry["kas_ms"]:
                    entry["speedup"] = entry["dense_ms"] / entry["kas_ms"]
                timing[key] = entry
                rows.append([
                    f"{V:,}", N,
                    f"{entry['dense_ms']:.2f}" if entry["dense_ms"] else "OOM",
                    f"{entry['kas_ms']:.2f}" if entry["kas_ms"] else "OOM",
                    f"{entry.get('speedup', float('nan')):.2f}x"
                    if entry.get("speedup") else "-",
                ])
                del h
                if device == "cuda":
                    torch.cuda.empty_cache()
            del head
            if device == "cuda":
                torch.cuda.empty_cache()
        table(rows, ["vocab", "N", "dense ms", "KAS ms", "KAS speedup"])
    payload["timing"] = timing

    # ------------------------------------------- the loop trap, measured
    if torch.cuda.is_available():
        rule("E3.3 -- trap 4: the obvious loop is latency-bound, not compute-bound")
        packed = PackedVocab(base, L_max=L_MAX)
        head = KASHead(D_MODEL, codec, packed, dtype=torch.float32).to("cuda")
        h = torch.randn(256, D_MODEL, device="cuda")
        W = torch.randn(packed.V, D_MODEL, device="cuda", dtype=torch.bfloat16)
        loop_ms = bench(lambda: head(h, impl="loop"), iters=5, device="cuda")
        bag_ms = bench(lambda: head(h, impl="bag"), device="cuda")
        dense_ms = bench(lambda: h.to(torch.bfloat16) @ W.t(), device="cuda")
        rows = [
            ["dense matmul", f"{dense_ms:.2f}", "1.00x"],
            [f"KAS slice-add loop ({L_MAX} kernel launches)",
             f"{loop_ms:.2f}", f"{dense_ms / loop_ms:.2f}x"],
            ["KAS as a sparse matmul (embedding_bag)",
             f"{bag_ms:.2f}", f"{dense_ms / bag_ms:.2f}x"],
        ]
        table(rows, ["implementation", "ms @ N=256, V=50k", "vs dense"])
        payload["loop_trap"] = {
            "dense_ms": dense_ms, "loop_ms": loop_ms, "bag_ms": bag_ms,
            "loop_to_bag_speedup": loop_ms / bag_ms,
        }
        del head, h, W
        torch.cuda.empty_cache()

    write_json("e3_cost.json", payload)

    rule("E3 summary")
    print("defensible:  parameters, optimizer memory, exactness, CPU / small-batch.")
    print("not claimed: a GPU wall-clock speedup proportional to the op ratio.")


if __name__ == "__main__":
    main()
