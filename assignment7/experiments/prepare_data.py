"""Tokenize a corpus into uint32 bins, in packed-vocabulary order.

    python experiments/prepare_data.py --source fineweb --tokens 25000000
"""

from __future__ import annotations

import argparse
import time

from _common import rule
from kronecker_v2.data import DEFAULT_DATA_DIR, build_bins
from kronecker_v2.vocab import PackedVocab, gpt2_token_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fineweb")
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    rule(f"tokenizing {args.source} -> {args.tokens:,} training tokens")
    packed = PackedVocab(gpt2_token_bytes(), L_max=128)
    t0 = time.perf_counter()
    meta = build_bins(
        args.source, packed, args.tokens, data_dir=args.data_dir, tag=args.tag
    )
    dt = time.perf_counter() - t0
    print(f"train tokens : {meta['n_train']:,}")
    print(f"val tokens   : {meta['n_val']:,} (first {meta['n_val_docs']} documents)")
    print(f"documents    : {meta['n_docs']:,}")
    print(f"elapsed      : {dt / 60:.1f} min")
    print(f"note         : {meta['note']}")


if __name__ == "__main__":
    main()
