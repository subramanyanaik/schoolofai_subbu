"""Diff an independent Colab run against the committed local run, and keep the README honest.

results/results.json is the run this repo committed (local RTX 3050).  results/colab_summary.json
is extracted by tools/extract_colab.py from an independent Colab run of the same notebook, and
results/colab_run_log.txt is that run's full stdout.

Reproducibility here has a shape, and the shape is the claim:

* Some quantities are *provably* machine-independent -- integer counts, parameter arithmetic,
  and allocator bytes, which are driven by tensor shapes rather than by the card.  Those must
  be identical, and CI fails if they are not, because a difference there is a bug and not
  noise.
* Everything downstream of GPU float work may only agree.  That includes numbers which in
  fact came out identical: item 5's initialisation anchor is a forward pass on the device, so
  its agreement is an observation, not a guarantee, and enforcing it would make CI fail on
  legitimate noise later.  Those are reported as "identical in this pair" without being
  enforced.
* Two runs cannot be compared at all where the fixture itself is drawn from the CUDA RNG.
  Item 7's correctness losses come from `torch.randn(..., device=device)`, so the two runs are
  measuring different random tensors.  The real invariant there is *within* a run -- every
  chunk size must produce the same loss -- and that is checked per-run instead.

    python tools/compare_colab.py --print
    python tools/compare_colab.py --write    # inject the table into README.md
    python tools/compare_colab.py --check    # exit 1 on README drift, on a must-match
                                             # violation, or on a flat-sweep violation
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

# Provably machine-independent. Enforced.
MUST_MATCH = {
    "item1_hidden_numel": "a product of shapes",
    "item1_logits_numel": "a product of shapes",
    "item1_blowup": "V/D",
    "item3_tokens_counted": "an integer count",
    "item3_tokens_masked": "an integer count",
    "item3_pad_fraction_pct": "a ratio of integer counts",
    "item4_n_before": "an integer count",
    "item4_n_after": "an integer count",
    "item6_body_params": "parameter arithmetic",
    "item6_embed_params": "parameter arithmetic",
    "item6_untied_total": "parameter arithmetic",
    "item6_tied_total": "parameter arithmetic",
    "item6_saved": "parameter arithmetic",
    "item6_saved_pct": "parameter arithmetic",
    "item7_peak_naive_mib": "allocator bytes, driven by tensor shapes not by the card",
    "item7_peak_chunked_mib": "allocator bytes, driven by tensor shapes not by the card",
    "item7_ratio": "allocator bytes, driven by tensor shapes not by the card",
}

# Downstream of GPU float work. Compared, not enforced beyond a loose sanity bound.
TOLERANCE = 0.05

# Quantities that are *differences* of two nearly-equal losses have no meaningful relative
# scale: part2_head1_delta_vs_baseline is ~0.05 nats formed from two numbers near 4.8, so a
# 0.006 wobble in the operands -- well inside the noise those operands show individually --
# reads as 11% of the result. Compare these on an absolute scale, in nats, instead.
ABSOLUTE_NATS = {
    "part2_head1_delta_vs_baseline": 0.02,
    "part2_gap": 0.02,
    "item3b_gap": 0.05,
}

# Drawn from the CUDA RNG, so the two runs are not measuring the same tensors at all.
INCOMPARABLE = {
    "item7_loss_absdiff": "the fixture is torch.randn(..., device=device); the two runs hold "
                          "different random tensors, so only the within-run invariant means "
                          "anything",
}

SWEEP_LOSS_SPREAD = 1e-4      # every chunk size must agree, within a run


def load():
    local = json.load(io.open(LOCAL, encoding="utf-8"))
    colab = json.load(io.open(COLAB, encoding="utf-8"))
    return local, colab


def scalar_rows(local, colab):
    rows = []
    for k, v in colab.items():
        if k.startswith("_") or k.startswith("env_") or k == "item7_sweep":
            continue
        if k in INCOMPARABLE or k not in local:
            continue
        mine = local[k]
        delta = v - mine
        rel = abs(delta) / max(abs(mine), 1e-12)
        rows.append((k, mine, v, delta, rel, k in MUST_MATCH))
    return rows


def sweep_check(name, sweep):
    losses = [s["loss"] for s in sweep]
    spread = max(losses) - min(losses)
    return spread, spread <= SWEEP_LOSS_SPREAD


def verify(rows, local, colab):
    problems = []
    for k, mine, theirs, delta, rel, must in rows:
        if must and delta != 0:
            problems.append(f"{k} must be identical across machines ({MUST_MATCH[k]}) "
                            f"but local={mine} colab={theirs}")
        elif not must and k in ABSOLUTE_NATS:
            if abs(delta) > ABSOLUTE_NATS[k]:
                problems.append(f"{k} differs by {abs(delta):.4f} nats, beyond the "
                                f"{ABSOLUTE_NATS[k]} nat bound for a difference of two losses "
                                f"(local={mine} colab={theirs})")
        elif not must and rel > TOLERANCE:
            problems.append(f"{k} differs by {rel * 100:.2f}%, beyond the "
                            f"{TOLERANCE * 100:.0f}% tolerance (local={mine} colab={theirs})")

    for name, sweep in (("local", local.get("item7_sweep")), ("colab", colab.get("item7_sweep"))):
        if not sweep:
            problems.append(f"{name} run has no item7_sweep to check")
            continue
        spread, ok = sweep_check(name, sweep)
        if not ok:
            problems.append(f"{name} chunk sweep is not flat: losses span {spread:.2e}, "
                            f"so chunking changed the objective")

    lp = {s["chunk"]: s["peak_mib"] for s in local.get("item7_sweep", [])}
    cp = {s["chunk"]: s["peak_mib"] for s in colab.get("item7_sweep", [])}
    for chunk in sorted(set(lp) & set(cp)):
        if lp[chunk] != cp[chunk]:
            problems.append(f"sweep peak at chunk={chunk} must be identical (allocator bytes) "
                            f"but local={lp[chunk]} colab={cp[chunk]}")
    return problems


def render(local, colab, rows):
    must = [r for r in rows if r[5]]
    soft = sorted((r for r in rows if not r[5]), key=lambda r: -r[4])
    same_anyway = [r for r in soft if r[3] == 0]
    drifted = [r for r in soft if r[3] != 0]
    lp = {s["chunk"]: s["peak_mib"] for s in local["item7_sweep"]}
    cp = {s["chunk"]: s["peak_mib"] for s in colab["item7_sweep"]}
    shared = sorted(set(lp) & set(cp))

    L = []
    a = L.append
    a("**Independently reproduced on Google Colab.** The run is extracted from its executed "
      "notebook by [`tools/extract_colab.py`](tools/extract_colab.py), which refuses the "
      "notebook unless its code cells are byte-identical to the committed one and it ran "
      "`execution_count` 1…22 with no gaps and no errors. Full stdout: "
      "[`results/colab_run_log.txt`](results/colab_run_log.txt).")
    a("")
    a("It is not the same machine in any respect that matters:")
    a("")
    a("| | this repo's committed run | the Colab run |")
    a("|---|---|---|")
    a(f"| GPU | {local.get('gpu')} ({local.get('gpu_total_gib')} GiB) | "
      f"{colab['env_gpu']} ({colab['env_gpu_gib']} GiB) |")
    a(f"| torch | {local['torch_version']} | {colab['env_torch']} |")
    a(f"| Python | 3.12.6 | {colab['env_python']} |")
    a("")
    a("A fresh Colab VM also ships without `tiktoken` and without the corpus, so that run "
      "exercised the cold `pip install` and the corpus download the local execution could "
      "only simulate.")
    a("")
    a(f"### {len(must)} quantities that *must* be identical, and are")
    a("")
    a("Different card, different CUDA, a torch six minor releases apart, a different Python — "
      "and these do not move, because nothing about them is a float computed on the device:")
    a("")
    a("| quantity | both runs | why it cannot differ |")
    a("|---|---|---|")
    for k, mine, _t, _d, _r, _m in must:
        a(f"| `{k}` | {mine:,} | {MUST_MATCH[k]} |")
    a("")
    a(f"Item 7's whole chunk sweep agrees too — peak memory at every chunk size "
      f"({', '.join(str(c) for c in shared)}) is identical across the two machines: "
      f"{', '.join(f'{lp[c]:,.1f}' for c in shared)} MiB. **The {local['item7_ratio']}× is a "
      "property of the algorithm, not of my card.** `max_memory_allocated` counts allocator "
      "bytes, and those are decided by tensor shapes.")
    a("")
    a(f"### {len(drifted)} quantities downstream of GPU training, which only agree")
    a("")
    a("These sit behind hundreds of GPU training steps and are *not* enforced, only compared. "
      "Largest disagreement first:")
    a("")
    a("| quantity | local | Colab | delta | relative |")
    a("|---|---|---|---|---|")
    for k, mine, theirs, delta, rel, _m in drifted:
        a(f"| `{k}` | {mine} | {theirs} | {delta:+.4f} | {rel * 100:.2f}% |")
    a("")
    if same_anyway:
        a(f"A further {len(same_anyway)} came out identical without being required to — "
          + ", ".join(f"`{r[0]}`" for r in same_anyway) + ". Item 5's anchor is among them, "
          "because parameters are seeded on the CPU RNG before the model reaches the device. "
          "That is an **observation, not a guarantee**: it is still a forward pass on the GPU, "
          "so it is reported rather than enforced, and CI will not fail if a later run moves "
          "it in the fourth decimal.")
        a("")
    d = dict((r[0], r) for r in drifted)
    bt, dv = d["item4_boundary_term"], d["part2_head1_delta_vs_baseline"]
    a(f"The two largest entries are both the same lesson about what a percentage means.")
    a("")
    a(f"`part2_head1_delta_vs_baseline` heads the list at {dv[4] * 100:.1f}%, but it is a "
      f"*difference of two nearly-equal losses* — about {abs(dv[1]):.2f} nats formed from two "
      "numbers near 4.8. Its operands each agree to a thousandth, and that wobble is most of "
      f"the {abs(dv[3]):.4f} nat gap. It has no meaningful relative scale, so the checker "
      f"compares it in nats instead. The result itself reproduces where it counts: the extra "
      f"head cost head 1 **{local['part2_head1_delta_vs_baseline']:+.4f}** nats locally and "
      f"**{colab['part2_head1_delta_vs_baseline']:+.4f}** on Colab — same sign, same order. "
      "Same seed, though, so that is confirmation the finding is not an artefact of my "
      "hardware, not a second sample.")
    a("")
    a(f"`item4_boundary_term` at {bt[4] * 100:.2f}% is the genuinely noise-prone one: a single "
      "prediction site, on one sequence, under a model that has taken 600 GPU steps. Which is "
      "exactly why item 4 does not rest on it. The 256-pair aggregate it uses instead lands at "
      f"**{local['item4_mean_ratio']}×** locally against **{colab['item4_mean_ratio']}×** on "
      f"Colab, with the boundary the worse site in **{local['item4_boundary_worse_pct']:.0f}%** "
      f"of pairs against **{colab['item4_boundary_worse_pct']:.0f}%**. The robust statistic "
      "reproduces; the single draw does not. That is the case for having built it.")
    a("")
    earlier = os.path.join(HERE, "results", "colab_earlier_partial.json")
    if os.path.exists(earlier):
        e = {k: v for k, v in json.load(io.open(earlier, encoding="utf-8")).items()
             if not k.startswith("_")}
        exact = [k for k in e if k in colab and e[k] == colab[k]]
        moved = {k: abs(e[k] - colab[k]) for k in e if k in colab and e[k] != colab[k]}
        a("### A third run, which sharpens the diagnosis")
        a("")
        a("An earlier Colab run of the same notebook was reported before its notebook was "
          "available, as a pasted summary. It is hand-transcribed into "
          "[`results/colab_earlier_partial.json`](results/colab_earlier_partial.json), is not "
          "machine-checkable, and no gate depends on it — but it answers a question the "
          "two-run comparison cannot.")
        a("")
        a(f"Of the {len(e)} quantities it covers, **{len(exact)} are bit-identical between the "
          "two Colab runs**, including all of item 3b — which nonetheless differs from the "
          "local run. Meanwhile the longer-running quantities differ between the two Colab "
          f"runs too, by up to {max(moved.values()):.4f}.")
        a("")
        a("So the drift is not simply \"different hardware\". Item 3b (400 steps at T=128) is "
          "reproducible on a fixed machine and shifts across machines, which is a systematic "
          "kernel difference. The reference model (600 steps at T=256) and Part 2 (800 steps) "
          "do not even reproduce Colab-to-Colab, which is nondeterminism inside a run being "
          "amplified by training length. The honest summary is that **agreement decays with "
          "the number of training steps behind a number**, and every quantity in the enforced "
          "table above has zero of them.")
        a("")
    a("### What is not comparable")
    a("")
    for k, why in INCOMPARABLE.items():
        a(f"* `{k}` — {why}. Within each run the sweep is flat instead, and that *is* enforced: "
          f"local losses span {max(s['loss'] for s in local['item7_sweep']) - min(s['loss'] for s in local['item7_sweep']):.1e} "
          f"across all six chunk sizes, Colab's {max(s['loss'] for s in colab['item7_sweep']) - min(s['loss'] for s in colab['item7_sweep']):.1e}.")
    a("")
    a("*This section is generated from [`results/colab_summary.json`](results/colab_summary.json) "
      "by [`tools/compare_colab.py`](tools/compare_colab.py), which fails if a must-match "
      "quantity is not identical or if either run's chunk sweep is not flat.*")
    return "\n".join(L)


def main():
    local, colab = load()
    rows = scalar_rows(local, colab)
    problems = verify(rows, local, colab)
    for p in problems:
        print("FAIL:", p)
    if problems:
        return 1

    block = render(local, colab, rows)
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
              f"({len([r for r in rows if r[5]])} enforced identical, "
              f"{len([r for r in rows if not r[5]])} compared)")
        return 0
    io.open(README, "w", encoding="utf-8", newline="\n").write(
        text[:i + len(BEGIN)] + "\n\n" + block.strip() + "\n\n" + text[j:])
    print("README.md Colab section updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
