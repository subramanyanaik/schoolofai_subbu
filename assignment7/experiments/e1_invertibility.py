"""E1 -- invertibility.  A hardware-independent acceptance gate.

Requirements (README III.6):
  * 100% exact on all 50,257 GPT-2 tokens at L_max = 128
  * 100% on 256,000 random byte strings, d_c in {16, 32, 64}
  * 100% on real multi-script UTF-8
  * at matched D = 4096: V1 d_p=16 loses tokens, V2 d_c=32 L_max=128 loses none
"""

from __future__ import annotations

import time

import numpy as np

from _common import rule, table, write_json
from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.vocab import gpt2_token_bytes

MULTISCRIPT = [
    "apple", "Hello, world!", "counterrevolutionaries",
    "भारत एक विशाल देश है",            # Devanagari
    "తెలుగు భాష చాలా అందమైనది",       # Telugu
    "中文分词测试用例",                 # CJK
    "Кириллица работает",              # Cyrillic
    "עברית מימין לשמאל",               # Hebrew
    "العربية لغة جميلة",               # Arabic
    "\U0001f37a\U0001f680\U0001f9ee",  # emoji
    "mixed भारत 中文 \U0001f37a end",
]

CONFIGS = [
    ("V1 d_p=16", dict(variant="v1", L_max=16)),
    ("V1 d_p=32", dict(variant="v1", L_max=32)),
    ("V2 d_c=32, L=128", dict(d_c=32, L_max=128)),
    ("V2 d_c=64, L=128", dict(d_c=64, L_max=128)),
]


def random_strings(n: int, lo: int, hi: int, seed: int) -> list[bytes]:
    rng = np.random.default_rng(seed)
    lens = rng.integers(lo, hi + 1, size=n)
    flat = rng.integers(0, 256, size=int(lens.sum()), dtype=np.uint8).tobytes()
    out, pos = [], 0
    for L in lens:
        out.append(flat[pos : pos + int(L)])
        pos += int(L)
    return out


def main() -> None:
    payload: dict = {}

    # ------------------------------------------------ the GPT-2 vocabulary
    rule("E1.1 -- GPT-2 vocabulary round-trip (50,257 surface forms)")
    toks = gpt2_token_bytes()
    rows, vocab = [], {}
    for name, kw in CONFIGS:
        codec = KroneckerCodec(**kw)
        t0 = time.perf_counter()
        st = codec.roundtrip_stats(toks, chunk=2048)
        st["seconds"] = round(time.perf_counter() - t0, 2)
        st["D"] = codec.D
        vocab[name] = st
        rows.append([
            name, codec.D,
            f"{st['exact']:,} ({100 * st['exact_frac']:.3f}%)",
            f"{100 * st['exact_frac_within_L_max']:.3f}%",
            f"{len(toks) - st['within_L_max']:,}",
            st["seconds"],
        ])
    table(rows, ["codec", "D", "exact / 50,257", "within L_max", "truncated", "s"])
    payload["gpt2_vocab"] = vocab

    lens = np.array([len(t) for t in toks])
    payload["vocab_stats"] = {
        "n": len(toks),
        "mean_len": float(lens.mean()),
        "max_len": int(lens.max()),
        "p99_len": float(np.percentile(lens, 99)),
    }
    print(f"\nmean token length {lens.mean():.2f} bytes, max {lens.max()}, "
          f"p99 {np.percentile(lens, 99):.0f}")

    # ---------------------------------------------------- matched-D compare
    rule("E1.2 -- matched D = 4096: what V1 loses and V2 does not")
    m_rows = []
    for name in ("V1 d_p=16", "V2 d_c=32, L=128"):
        st = vocab[name]
        m_rows.append([name, st["D"], len(toks) - st["within_L_max"],
                       f"{100 * st['exact_frac']:.3f}%"])
    table(m_rows, ["codec", "D", "tokens truncated", "exact"])
    payload["matched_D_4096"] = {
        "v1_dp16_truncated": len(toks) - vocab["V1 d_p=16"]["within_L_max"],
        "v2_dc32_L128_truncated": len(toks) - vocab["V2 d_c=32, L=128"]["within_L_max"],
        "length_limit_ratio": 128 / 16,
    }

    # ------------------------------------------------------- random strings
    rule("E1.3 -- 256,000 random byte strings per d_c")
    rand_rows, rand = [], {}
    for d_c in (16, 32, 64):
        codec = KroneckerCodec(d_c=d_c, L_max=64)
        seqs = random_strings(256_000, 1, 64, seed=d_c)
        t0 = time.perf_counter()
        st = codec.roundtrip_stats(seqs, chunk=4096)
        st["seconds"] = round(time.perf_counter() - t0, 2)
        rand[f"d_c={d_c}"] = st
        rand_rows.append([d_c, codec.D, f"{st['exact']:,} / {st['n']:,}",
                          f"{100 * st['exact_frac']:.4f}%", st["seconds"]])
    table(rand_rows, ["d_c", "D", "exact", "rate", "s"])
    payload["random_strings"] = rand

    # -------------------------------------------------------- multi-script
    rule("E1.4 -- real multi-script UTF-8")
    codec = KroneckerCodec(d_c=32, L_max=128)
    ms_rows, ms = [], []
    for s in MULTISCRIPT:
        b = s.encode("utf-8")
        got = codec.roundtrip([b])[0]
        ms.append({"text": s, "bytes": len(b), "ok": got == b})
        ms_rows.append([s[:34], len(b), "OK" if got == b else "FAIL"])
    table(ms_rows, ["text", "bytes", "roundtrip"])
    payload["multiscript"] = ms

    # ---------------------------------------------- truncate vs alias (E1.5)
    rule("E1.5 -- overflow: truncation vs aliasing (L_max = 64)")
    ov_rows, overflow = [], {}
    for L in (72, 96, 128):
        seqs = random_strings(512, L, L, seed=L)
        entry = {}
        for mode in ("truncate", "alias"):
            c = KroneckerCodec(d_c=32, L_max=64, overflow=mode)
            entry[mode] = c.roundtrip_stats(seqs, chunk=512)["byte_accuracy"]
        overflow[str(L)] = entry
        ov_rows.append([L, f"{entry['truncate']:.3f}", f"{entry['alias']:.3f}"])
    table(ov_rows, ["token length", "truncate", "alias"])
    print("\naliasing corrupts every position at once; truncation keeps a correct")
    print("prefix.  the original design specified aliasing; measurement killed it.")
    payload["overflow"] = overflow

    # ---------------------------------------------------------- decode agree
    rule("E1.6 -- sign read == 256-way nearest neighbour")
    c = KroneckerCodec(d_c=32, L_max=64)
    seqs = random_strings(20_000, 1, 64, seed=99)
    K = c.encode(seqs)
    agree = c.decode(K, method="sign") == c.decode(K, method="nn")
    print(f"agreement on 20,000 strings: {agree}")
    payload["sign_equals_nn"] = bool(agree)

    # ------------------------------------------------------------ the gates
    rule("E1 gates")
    gates = {
        "gpt2_100pct_at_L128_dc32": vocab["V2 d_c=32, L=128"]["exact_frac"] == 1.0,
        "gpt2_100pct_at_L128_dc64": vocab["V2 d_c=64, L=128"]["exact_frac"] == 1.0,
        "random_100pct": all(v["exact_frac"] == 1.0 for v in rand.values()),
        "multiscript_100pct": all(m["ok"] for m in ms),
        "v1_dp16_loses_tokens_at_D4096":
            payload["matched_D_4096"]["v1_dp16_truncated"] > 0,
        "v2_loses_none_at_D4096":
            payload["matched_D_4096"]["v2_dc32_L128_truncated"] == 0,
        "truncate_beats_alias": all(
            v["truncate"] > v["alias"] for v in overflow.values()
        ),
        "sign_equals_nn": bool(agree),
    }
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    payload["gates"] = gates
    payload["all_gates_pass"] = all(gates.values())
    write_json("e1_invertibility.json", payload)
    if not all(gates.values()):
        raise SystemExit("E1 gate failed")


if __name__ == "__main__":
    main()
