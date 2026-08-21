"""Structural checks on the attention timeline.

The assignment's whole risk is a confident, wrong date and a technique written
up with only upsides. These checks make both of those failures loud instead of
silent, so they cannot survive a push.

    python assignment8/tools/validate.py
"""
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data.js"
SITE = ROOT / "site"

REQUIRED = ["name", "abbr", "date", "dateLabel", "who", "src", "family", "era",
            "headline", "problem", "idea", "math", "pros", "cons", "pick"]
FAMILIES = {"exact", "pos", "kv", "sparse", "linear", "systems"}

failures = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


def blocks(src):
    return re.split(r"\n\{\n  id: ", src)[1:]


def field(block, key):
    m = re.search(r'[\s,]' + key + r': "((?:[^"\\]|\\.)*)"', block)
    return m.group(1) if m else None


def listlen(block, key):
    m = re.search(key + r": \[(.*?)\n  \]", block, re.S)
    if not m:
        m = re.search(key + r": \[(.*?)\],\n", block, re.S)
    return len(re.findall(r'^\s*"', m.group(1), re.M)) if m else 0


def main():
    src = DATA.read_text(encoding="utf-8")
    bs = blocks(src)
    check(len(bs) >= 30, f"expected 30+ mechanisms, found {len(bs)}")

    dates, ids = [], []
    for b in bs:
        mid = b.split('"')[1]
        ids.append(mid)

        for key in REQUIRED:
            check(re.search(r'[\s,]' + key + r': ', b) is not None,
                  f"{mid}: missing required field '{key}'")

        d = field(b, "date")
        check(d is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d or ""),
              f"{mid}: date is not an ISO yyyy-mm-dd value (got {d!r})")
        if d:
            dates.append(d)
            check("2014-01-01" <= d <= "2027-12-31", f"{mid}: date {d} is outside a plausible range")

        fam = field(b, "family")
        check(fam in FAMILIES, f"{mid}: unknown family {fam!r}")

        # every date must be defensible: at least one primary source link
        srcm = re.search(r"src: \[(.*?)\],\n", b, re.S)
        urls = re.findall(r'url:"([^"]*)"', srcm.group(1)) if srcm else []
        check(len(urls) >= 1, f"{mid}: no source link for its date")
        for u in urls:
            check(u.startswith("http"), f"{mid}: malformed source url {u!r}")

        # the honesty rule: nothing gets written up with only upsides
        npros, ncons = listlen(b, "pros"), listlen(b, "cons")
        check(npros >= 3, f"{mid}: only {npros} pros — thin")
        check(ncons >= 3, f"{mid}: only {ncons} cons — a technique with fewer than "
                          f"three real downsides has not been understood yet")

    check(dates == sorted(dates),
          "MECHANISMS is not in chronological order — the timeline's entire claim")
    check(len(set(ids)) == len(ids), "duplicate mechanism ids")

    # DATES.md must be regenerated whenever data.js changes
    out = ROOT / "DATES.md"
    check(out.exists(), "DATES.md is missing — run tools/gen_date_table.py")
    if out.exists():
        txt = out.read_text(encoding="utf-8")
        check(f"**{len(bs)} mechanisms.**" in txt,
              f"DATES.md is stale ({len(bs)} mechanisms in data.js) — "
              f"run tools/gen_date_table.py")
        for b in bs:
            ab = field(b, "abbr")
            if ab:
                check(f"**{ab}**" in txt or f"**{ab.replace('|', '')}**" in txt,
                      f"DATES.md has no row for {ab!r} — run tools/gen_date_table.py")

    # the site must stay dependency-free and offline
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for tag in re.findall(r'<(?:script|link)[^>]*>', html):
        m = re.search(r'(?:src|href)="([^"]+)"', tag)
        if m and re.match(r"https?://", m.group(1)):
            check(False, f"external asset would break the offline claim: {m.group(1)}")
    check("<script src=\"data.js\">" in html, "index.html does not load data.js")

    print(f"{checks - len(failures)}/{checks} checks passed "
          f"({len(bs)} mechanisms, {len(set(ids))} unique ids)")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
