"""Diff an independent Colab run against the committed local run, and keep the README honest.

results/results.json is the run this repo committed (local RTX 3050).  results/colab_summary.json
is a transcription of the final SUMMARY cell from an independent Colab GPU run, and
results/colab_summary.txt is the verbatim paste that transcription came from.

Reproducibility here is a claim with a shape: quantities fixed by shapes, integer counts,
CPU-seeded initialisation or pure arithmetic must come out *identical* on different hardware,
and quantities downstream of hundreds of GPU training steps must come out *close*.  Anything
that violates that split is a bug, not noise -- so the split is asserted rather than eyeballed.

    python tools/compare_colab.py --print
    python tools/compare_colab.py --write    # inject the table into README.md
    python tools/compare_colab.py --check    # exit 1 if README.md has drifted, or if a
                                             # should-be-identical number is not identical
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(HERE, "results", "results.json")
COLAB = os.path.join(HERE, "results", "colab_summary.json")
README = os.path.join(HERE, "README.md")
BEGIN, END = "<!-- BEGIN:colab -->", "<!-- END:colab -->"

# Quantities that must be bit-identical on any machine, and why.
MUST_MATCH = {
    "item3_tokens_counted": "an integer count",
    "item3_tokens_masked": "an integer count",
    "item3_pad_fraction_pct": "a ratio of integer counts",
    "item4_n_before": "an integer count",
    "item4_n_after": "an integer count",
    "item5_init_ppl": "one forward pass of a CPU-seeded initialisation",
    "item5_init_loss": "one forward pass of a CPU-seeded initialisation",
    "item5_ppl_over_vocab": "one forward pass of a CPU-seeded initialisation",
    "item6_untied_total": "parameter arithmetic",
    "item6_tied_total": "parameter arithmetic",
    "item6_saved": "parameter arithmetic",
    "item6_saved_pct": "parameter arithmetic",
    "item7_peak_naive_mib": "allocator bytes, driven by tensor shapes not by the card",
    "item7_peak_chunked_mib": "allocator bytes, driven by tensor shapes not by the card",
    "item7_ratio": "allocator bytes, driven by tensor shapes not by the card",
    "part3_copy_rate_pct": "a saturated rate",
}

# Everything else sits downstream of GPU training, so it may only be close.
TOLERANCE = 0.05          # 5% relative; the loosest observed is item4_boundary_term


def load():
    local = json.load(io.open(LOCAL, encoding="utf-8"))
    colab = {k: v for k, v in json.load(io.open(COLAB, encoding="utf-8")).items()
             if not k.startswith("_")}
    missing = [k for k in colab if k not in local]
    if missing:
        raise SystemExit(f"colab_summary.json has keys results.json does not: {missing}")
    return local, colab


def rows(local, colab):
    out = []
    for k, v in colab.items():
        mine = local[k]
        delta = v - mine
        rel = abs(delta) / max(abs(mine), 1e-12)
        out.append((k, mine, v, delta, rel, k in MUST_MATCH))
    return out


def verify(data):
    """The split is the claim; a violation of it is a bug, not noise."""
    problems = []
    for k, mine, theirs, delta, rel, must in data:
        if must and delta != 0:
            problems.append(f"{k} must be identical across machines ({MUST_MATCH[k]}) "
                            f"but local={mine} colab={theirs}")
        if not must and rel > TOLERANCE:
            problems.append(f"{k} differs by {rel * 100:.2f}%, beyond the "
                            f"{TOLERANCE * 100:.0f}% float-noise tolerance "
                            f"(local={mine} colab={theirs})")
    return problems


def render(data):
    ident = [d for d in data if d[5]]
    noisy = sorted((d for d in data if not d[5]), key=lambda d: -d[4])
    L = []
    a = L.append
    a("**Independently reproduced on a Google Colab GPU runtime.** A fresh Colab VM ships "
      "without `tiktoken` and without the corpus, so that run also exercised the cold "
      "`pip install` and the corpus download that the local execution could not.")
    a("")
    a(f"**{len(ident)} of the {len(data)} reported quantities came out identical** — every one "
      "that is fixed by tensor shapes, integer counts, CPU-seeded initialisation or plain "
      "arithmetic, and therefore *must* be:")
    a("")
    a("| quantity | both runs | why it cannot differ |")
    a("|---|---|---|")
    for k, mine, _t, _d, _r, _m in ident:
        val = f"{mine:,}" if isinstance(mine, int) else f"{mine:,}"
        a(f"| `{k}` | {val} | {MUST_MATCH[k]} |")
    a("")
    a(f"The other {len(noisy)} sit downstream of hundreds of GPU training steps, so they may "
      "only agree, not match. Largest disagreement first:")
    a("")
    a("| quantity | local (RTX 3050) | Colab | delta | relative |")
    a("|---|---|---|---|---|")
    for k, mine, theirs, delta, rel, _m in noisy:
        a(f"| `{k}` | {mine} | {theirs} | {delta:+.4f} | {rel * 100:.2f}% |")
    a("")
    worst = noisy[0]
    a(f"The worst of them, `{worst[0]}` at {worst[4] * 100:.2f}%, is the single most "
      "noise-prone number in the whole harness: one prediction site, on one sequence, under a "
      "model that has taken 600 GPU training steps. It is exactly why item 4 does not rest on "
      "it, and repeats the measurement over 256 packed pairs instead.")
    a("")
    a("Two results are worth pulling out. Item 5's anchor is **identical** because parameter "
      "initialisation runs on the CPU RNG before the model moves to the device, so the "
      "cheapest sanity check in the session is hardware-independent. And item 7's memory "
      "numbers are **identical** on two different GPUs, because `max_memory_allocated` counts "
      "allocator bytes driven by tensor shapes — the 11.6× is a property of the algorithm, "
      "not of my card.")
    a("")
    a("*Transcribed from [`results/colab_summary.txt`](results/colab_summary.txt) into "
      "[`results/colab_summary.json`](results/colab_summary.json); this table is generated "
      "from that JSON by [`tools/compare_colab.py`](tools/compare_colab.py), which also fails "
      "if a should-be-identical quantity is not identical. The Colab GPU model was not "
      "recorded.*")
    return "\n".join(L)


def main():
    local, colab = load()
    data = rows(local, colab)
    problems = verify(data)
    for p in problems:
        print("FAIL:", p)
    if problems:
        return 1

    block = render(data)
    if "--print" in sys.argv:
        print(block)
        return 0

    text = io.open(README, encoding="utf-8").read()
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < 0:
        print(f"README.md is missing the {BEGIN} / {END} markers")
        return 1
    if "--check" in sys.argv:
        if text[i + len(BEGIN):j].strip() != block.strip():
            print("README.md's Colab section disagrees with results/colab_summary.json.")
            print("Run: python tools/compare_colab.py --write")
            return 1
        print(f"README.md matches results/colab_summary.json "
              f"({len([d for d in data if d[5]])} identical, "
              f"{len([d for d in data if not d[5]])} within tolerance)")
        return 0
    io.open(README, "w", encoding="utf-8", newline="\n").write(
        text[:i + len(BEGIN)] + "\n\n" + block.strip() + "\n\n" + text[j:])
    print("README.md Colab section updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
