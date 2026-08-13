"""The property Problem 5 asks for, shown directly.

    "Kronecker is forward deterministic (same word will always give the same
     embedding). How do I make a reverse of this (same embedding gives the
     same Kronecker)?"

Same word -> same vector, and same vector -> same word, at matched dimension
where V1 loses the tail of the string.
"""

from __future__ import annotations

import numpy as np

from _common import rule, write_json
from kronecker_v2.codec import KroneckerCodec

CASES = [
    "apple",
    "a",
    "counterrevolutionaries",
    "भारत",
    "తెలుగు",
    "\U0001f37a",
    "antidisestablishmentarianism",
]


def show(s: str, codec: KroneckerCodec) -> tuple[str, bool]:
    got = codec.roundtrip([s.encode("utf-8")])[0]
    ok = got == s.encode("utf-8")
    try:
        shown = got.decode("utf-8")
    except UnicodeDecodeError:
        shown = got.decode("utf-8", errors="replace")
    return shown, ok


def main() -> None:
    v2 = KroneckerCodec(d_c=32, L_max=128)   # D = 4096
    v1 = KroneckerCodec(variant="v1", L_max=16)  # D = 4096
    assert v1.D == v2.D == 4096

    rule("Problem 5 demo -- reverse-deterministic codec, matched D = 4096")
    print(f"{'string':<32} {'bytes':>5}  {'V2 (D=4096)':<34} V1 (D=4096)")
    rows = []
    for s in CASES:
        n = len(s.encode("utf-8"))
        got2, ok2 = show(s, v2)
        got1, ok1 = show(s, v1)
        print(
            f"{s:<32} {n:>5}  "
            f"{('OK ' if ok2 else 'LOST ') + repr(got2):<34} "
            f"{('OK ' if ok1 else 'LOST ') + repr(got1)}"
        )
        rows.append(
            {"string": s, "bytes": n, "v2_ok": ok2, "v2": got2, "v1_ok": ok1, "v1": got1}
        )

    K = v2.encode(["apple".encode()])
    K2 = v2.encode(["apple".encode()])
    sign = v2.decode(K, method="sign")
    nn_ = v2.decode(K, method="nn")

    print()
    print(
        f"embedding is a plain float vector: shape {tuple(K.shape)}, "
        f"norm {np.linalg.norm(K):.6f}, no lookup table"
    )
    print(f"forward deterministic: {np.array_equal(K, K2)}    "
          f"reverse deterministic: {sign == nn_ == [b'apple']}")
    print(f"sign-read == nearest-neighbour decode: {sign == nn_}")
    print(f"decode cost: O(8) sign reads per byte, no {v2.G.shape[0]}-way search needed")

    write_json(
        "demo_invert.json",
        {
            "D": v2.D,
            "v2": repr(v2),
            "v1": repr(v1),
            "cases": rows,
            "forward_deterministic": bool(np.array_equal(K, K2)),
            "reverse_deterministic": bool(sign == nn_ == [b"apple"]),
            "embedding_norm": float(np.linalg.norm(K)),
        },
    )


if __name__ == "__main__":
    main()
