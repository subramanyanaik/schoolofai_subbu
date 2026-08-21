"""Measure the intervals and clustering in the timeline.

Question 2 asks what the date order shows that a list cannot. The honest way
to answer that is to measure it rather than assert it, so every interval
quoted in ANSWERS.md is computed here from site/data.js.

    python assignment8/tools/gaps.py
"""
import re
import datetime
import pathlib
from collections import Counter

DATA = pathlib.Path(__file__).resolve().parents[1] / "site" / "data.js"
FIELD = r'[\s,]{key}: "((?:[^"\\]|\\.)*)"'


def load():
    src = DATA.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n\{\n  id: ", src)[1:]:
        def get(key):
            m = re.search(FIELD.format(key=key), block)
            return m.group(1) if m else ""
        out.append({"date": get("date"), "abbr": get("abbr"),
                    "family": get("family"), "name": get("name")})
    return out


def main():
    rows = load()
    by = {r["abbr"]: r for r in rows}
    iso = datetime.date.fromisoformat

    def gap(a, b):
        days = (iso(by[b]["date"]) - iso(by[a]["date"])).days
        yrs = days / 365.25
        return (f"{a:<18} {by[a]['date']}  ->  {b:<18} {by[b]['date']}   "
                f"{days:>5} days  ({yrs:4.1f} yr)")

    print("INTERVALS THE LIST HIDES")
    print("-" * 78)
    for a, b in [("MQA", "GQA"),
                 ("Delta rule", "Parallel DeltaNet"),
                 ("Linear attention", "Gated DeltaNet"),
                 ("Top-k", "NSA"),
                 ("PI", "NTK-aware"),
                 ("NTK-aware", "YaRN"),
                 ("NSA", "MoBA"),
                 ("Learned APE", "Sinusoidal"),
                 ("RoPE", "DroPE")]:
        print(gap(a, b))

    print("\nDENSITY PER YEAR")
    print("-" * 78)
    c = Counter(r["date"][:4] for r in rows)
    for y in sorted(c):
        print(f"  {y}  {'#' * c[y]:<12} {c[y]}")

    print("\nWHEN EACH FAMILY WAS ACTIVE  (first -> last appearance)")
    print("-" * 78)
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r["date"])
    for f, ds in sorted(fam.items(), key=lambda kv: min(kv[1])):
        print(f"  {f:<9} {min(ds)}  ->  {max(ds)}   ({len(ds)} entries)")


if __name__ == "__main__":
    main()
