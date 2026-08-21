"""Regenerate the README date-source table straight from site/data.js.

The point of this script is that the README table and the live site can never
drift apart: both are generated from the same single source of truth. If a date
changes in data.js, re-run this and the README follows.

    python assignment8/tools/gen_date_table.py
"""
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data.js"
OUT = ROOT / "DATES.md"

FIELD = r'[\s,]{key}: "((?:[^"\\]|\\.)*)"'


def parse():
    src = DATA.read_text(encoding="utf-8")
    entries = []
    for block in re.split(r"\n\{\n  id: ", src)[1:]:
        def get(key):
            m = re.search(FIELD.format(key=key), block)
            return m.group(1) if m else ""

        srcm = re.search(r"src: \[(.*?)\],\n", block, re.S)
        srcs = re.findall(r'\{label:"((?:[^"\\]|\\.)*)", ?url:"([^"]*)"\}',
                          srcm.group(1)) if srcm else []
        entries.append({
            "date": get("date"),
            "label": get("dateLabel"),
            "abbr": get("abbr"),
            "name": get("name"),
            "who": get("who"),
            "srcs": srcs,
            "note": "dateNote:" in block,
            "covered": "covered: true" in block,
        })
    return entries


def clean(t):
    for a, b in [("&mdash;", "—"), ("&ndash;", "–"), ("§", "§"), ("|", "\\|"),
                 ("&times;", "×"), ("&eacute;", "é"), ("<em>", "*"), ("</em>", "*")]:
        t = t.replace(a, b)
    return t


def main():
    rows = parse()
    if not rows:
        sys.exit("no entries parsed — did data.js change shape?")
    if [r["date"] for r in rows] != sorted(r["date"] for r in rows):
        sys.exit("data.js is not in chronological order")

    out = [
        "# Every date on the timeline, and where it came from",
        "",
        "Generated from [`site/data.js`](site/data.js) by "
        "[`tools/gen_date_table.py`](tools/gen_date_table.py) — the site and this table "
        "cannot drift apart.",
        "",
        f"**{len(rows)} mechanisms.** "
        f"{sum(r['covered'] for r in rows)} were covered in Session 8; "
        f"{sum(not r['covered'] for r in rows)} are additions. "
        f"{sum(r['note'] for r in rows)} carry a date caveat, written out in full in the app.",
        "",
        "Dates are the **v1 arXiv submission timestamp** (not the announcement date, not the "
        "conference publication date, not the version you happen to be reading) or, for things "
        "that were never papers, the official release. Where those disagree, the disagreement "
        "is recorded rather than resolved silently.",
        "",
        "| Date (v1) | Mechanism | Origin | Source used for the date | Caveat |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        links = " <br> ".join(f"[{clean(l)}]({u})" for l, u in r["srcs"])
        out.append(
            f"| `{r['label']}` | **{clean(r['abbr'])}** | {clean(r['who'])} | {links} "
            f"| {'⚠ see the entry in the app' if r['note'] else '—'} |"
        )
    out.append("")
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT} — {len(rows)} rows, {sum(r['note'] for r in rows)} caveats")


if __name__ == "__main__":
    main()
