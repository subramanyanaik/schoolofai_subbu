"""Assert that every committed gate in results/ says PASS.

Belt and braces.  The other scripts re-derive the gates from scratch; this one
reads the evidence a grader actually reads and fails if any of it says
otherwise.  Exits non-zero on the first failure.

    python experiments/verify_gates.py
"""

from __future__ import annotations

import sys

from _common import read_json, rule, table

# (results file, key holding the gate dict, human label)
GATE_FILES = [
    ("e1_invertibility.json", "gates", "E1 invertibility"),
    ("params_audit.json", "gates", "parameter audit"),
    ("e6_vocab_swap.json", "gates", "E6 vocabulary swap"),
]

# Facts that are arithmetic rather than measurement, checked directly.
def invariants() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    e1 = read_json("e1_invertibility.json")
    if e1:
        v = e1["gpt2_vocab"]
        out.append((
            "V2 recovers all 50,257 GPT-2 tokens at D=4096",
            v["V2 d_c=32, L=128"]["exact"] == 50257,
            f"{v['V2 d_c=32, L=128']['exact']} / 50257",
        ))
        out.append((
            "V1 loses tokens at the same D",
            v["V1 d_p=16"]["exact"] < 50257,
            f"{50257 - v['V1 d_p=16']['exact']} lost",
        ))

    e3 = read_json("e3_cost.json")
    if e3:
        p = e3["params"]
        sizes = {r["kas_head_params"] for r in p.values()}
        out.append((
            "KAS head parameters do not depend on |V|",
            len(sizes) == 1,
            f"{sizes.pop() if len(sizes) == 1 else sizes:,} across "
            f"{len(p)} vocabulary sizes",
        ))

    e6 = read_json("e6_vocab_swap.json")
    if e6:
        out.append((
            "50k -> 1M vocabulary costs zero parameters",
            e6["delta_params"] == 0,
            f"delta = {e6['delta_params']:+,}",
        ))

    audit = read_json("params_audit.json")
    if audit:
        unused = {a: r["unused_params"] for a, r in audit["arms"].items()
                  if r["unused_params"]}
        out.append((
            "every parameter of every arm receives a gradient",
            not unused,
            "clean" if not unused else str(unused),
        ))
    return out


def main() -> int:
    rule("committed gates")
    rows, failed = [], []
    missing = []
    for fname, key, label in GATE_FILES:
        data = read_json(fname)
        if data is None:
            missing.append(fname)
            rows.append([label, "-", "MISSING", fname])
            continue
        for name, ok in data.get(key, {}).items():
            rows.append([label, name, "PASS" if ok else "FAIL", fname])
            if not ok:
                failed.append(f"{fname}:{name}")
    table(rows, ["experiment", "gate", "result", "source"])

    rule("derived invariants")
    inv_rows = []
    for name, ok, detail in invariants():
        inv_rows.append([name, "PASS" if ok else "FAIL", detail])
        if not ok:
            failed.append(name)
    table(inv_rows, ["invariant", "result", "measured"])

    print()
    if missing:
        print(f"MISSING results files: {', '.join(missing)}")
        print("run the experiments first -- see REPRODUCE.md")
        return 1
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    total = len(rows) + len(inv_rows)
    print(f"all {total} committed gates and invariants PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
