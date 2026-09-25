"""Render results/results.json into the <!-- BEGIN:numbers --> ... <!-- END:numbers --> block
in README.md, so every number in the write-up is read out of a real run's output rather than
typed in by hand.

    python tools/render_numbers.py            # rewrite README.md's numbers block
    python tools/render_numbers.py --check     # exit 1 if README.md disagrees with a fresh render

If results/results.json does not exist yet (the notebook has not been run on a GPU), this
writes a "pending" placeholder instead of failing, so the repo is committable before the
Colab run and self-updating after it.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(HERE, "README.md")
RESULTS = os.path.join(HERE, "results", "results.json")
BEGIN, END = "<!-- BEGIN:numbers -->", "<!-- END:numbers -->"


def render_pending():
    return (
        "\n> **Results pending.** These numbers come from a real training run on a Colab GPU "
        "(`reversible_llm.ipynb`), not from the machine this repo was assembled on, which "
        "cannot load torch's native libraries at all (see the note at the top of "
        "`reversible_llm.py` — confirmed WDAC block, not a missing package). Open the "
        "notebook in Colab with a GPU runtime, run it end to end, commit the `results/` "
        "folder it writes, then run `python tools/render_numbers.py` to fill in this section "
        "from that run.\n"
    )


def render_table(results):
    runs = results.get("runs", {})
    if not runs:
        return render_pending()
    lines = [
        f"**Device:** `{results.get('device', '?')}`  |  "
        f"**seed:** `{results.get('seed', '?')}`  |  "
        f"**winning reversible variant:** `{results.get('winner_variant', '?')}`",
        "",
        "| run | mode | batch size | params | steps | tokens trained | final loss | "
        "min loss | tokens/s | wall clock | peak memory |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        mem = f"{r['peak_memory_mib']:.0f} MiB" if r.get("peak_memory_mib") is not None else "n/a"
        lines.append(
            f"| `{name}` | {r['mode']} | {r['batch_size']:,} | {r['params']:,} | "
            f"{r['steps']:,} | {r['tokens_trained']:,} | {r['final_loss']:.4f} | "
            f"{r['min_loss']:.4f} | {r['tokens_per_s']:,.0f} | {r['wall_clock_s'] / 60:.1f} min "
            f"| {mem} |")
    lines.append("")

    if "baseline_max_batch" in results:
        lines.append(f"- baseline (`standard`) max batch size on this GPU: "
                      f"**{results['baseline_max_batch']:,}**")
    winner = results.get("winner_variant")
    if winner and f"{winner}_max_batch" in results:
        wmax = results[f"{winner}_max_batch"]
        lines.append(f"- `{winner}` (winning reversible variant) max batch size: **{wmax:,}**")
        base = results.get("baseline_max_batch")
        if base:
            lines.append(f"- that is **{wmax / base:.1f}x** the baseline's max batch size, "
                          f"on the same GPU")
    sizing = results.get("model_sizing", {})
    if sizing:
        lines.append("")
        lines.append("| backbone | n_layer | n_embd | parameters |")
        lines.append("|---|---|---|---|")
        for mode, s in sizing.items():
            lines.append(f"| `{mode}` | {s['n_layer']} | {s['n_embd']} | {s['params']:,} |")
    lines.append("")
    lines.append("![loss curves](results/loss_curves.png)")
    return "\n".join(lines) + "\n"


def render():
    if not os.path.exists(RESULTS):
        return render_pending()
    with io.open(RESULTS, encoding="utf-8") as fh:
        results = json.load(fh)
    return render_table(results)


_MARKER_RE = re.compile(
    r"^" + re.escape(BEGIN) + r"\n.*?\n" + re.escape(END) + r"$",
    re.MULTILINE | re.DOTALL)


def splice(readme_text, block):
    # Anchored to the *start of a line*, not a substring search, so a sentence that merely
    # mentions "<!-- BEGIN:numbers -->" in prose (as this file's own docstring might) can
    # never be mistaken for the real marker.
    if not _MARKER_RE.search(readme_text):
        raise SystemExit(f"README.md is missing a well-formed {BEGIN} ... {END} block "
                          f"(each marker must be alone on its own line)")
    return _MARKER_RE.sub(lambda _: f"{BEGIN}\n{block}{END}", readme_text, count=1)


def main():
    check = "--check" in sys.argv
    current = io.open(README, encoding="utf-8").read()
    updated = splice(current, render())
    if check:
        if current != updated:
            print("README.md's numbers block does not match results/results.json.")
            print("Run: python tools/render_numbers.py")
            return 1
        print("README.md numbers match results/results.json.")
        return 0
    if current != updated:
        io.open(README, "w", encoding="utf-8", newline="\n").write(updated)
        print(f"updated {README}")
    else:
        print("README.md already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
