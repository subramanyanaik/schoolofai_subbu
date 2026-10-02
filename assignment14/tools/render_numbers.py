"""Render results/results.json into the <!-- BEGIN:numbers --> ... <!-- END:numbers --> block
in README.md, so every number in the write-up is read out of a real run's output rather than
typed in by hand.

    python tools/render_numbers.py            # rewrite README.md's numbers block
    python tools/render_numbers.py --check     # exit 1 if README.md disagrees with a fresh render

If results/results.json does not exist yet, this writes a "pending" placeholder instead of
failing, so the repo is committable before a run and self-updating after one.
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
        "\n> **Results pending.** These numbers come from a real training run "
        "(`moe_llm.py` / `moe_llm.ipynb`). Run it, commit the `results/` folder it writes, "
        "then run `python tools/render_numbers.py` to fill in this section from that run.\n"
    )


def render_table(results):
    runs = results.get("runs", {})
    if not runs:
        return render_pending()
    pc = results.get("param_counts", {})
    cfg = results.get("model_config", {})
    lines = [
        f"**Device:** `{results.get('device', '?')}`  |  **seed:** `{results.get('seed', '?')}`  "
        f"|  **experts:** `{cfg.get('n_experts', '?')}`  |  **top-k:** `{cfg.get('top_k', '?')}`",
        "",
        "| run | starts from | total params | active params | tokens in this run | "
        "final train loss | final val loss | best val loss | tokens/s | wall clock | peak memory |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    totals = {"dense": pc.get("dense_total"), "moe": pc.get("moe_total"),
              "dense_continued": pc.get("dense_total")}
    actives = {"dense": pc.get("dense_total"), "moe": pc.get("moe_active"),
               "dense_continued": pc.get("dense_total")}
    starts = {"dense": "random init", "moe": "`dense`, upcycled",
              "dense_continued": "`dense`, unchanged"}
    for name, r in runs.items():
        mem = f"{r['peak_memory_mib']:.0f} MiB" if r.get("peak_memory_mib") is not None else "n/a"
        best_val = min(p["val_loss"] for p in r["val_curve"])
        lines.append(
            f"| `{name}` | {starts.get(name, '?')} | {totals[name]:,} | {actives[name]:,} | "
            f"{r['tokens_trained']:,} | {r['final_loss']:.4f} | {r['final_val_loss']:.4f} | "
            f"{best_val:.4f} | {r['tokens_per_s']:,.0f} | {r['wall_clock_s'] / 60:.1f} min | {mem} |")
    lines.append("")

    if pc:
        ratio_total = pc["moe_total"] / pc["dense_total"]
        ratio_active = pc["moe_active"] / pc["dense_total"]
        lines.append(f"- **Parameters:** the MoE has **{ratio_total:.2f}x** the dense model's total "
                     f"parameters but only **{ratio_active:.2f}x** its *active* parameters per "
                     f"token (**{pc['moe_active'] / pc['moe_total']:.1%}** of the MoE total).")
    seam = results.get("seam")
    if seam:
        lines.append(f"- **Cost of the conversion:** validation loss went from "
                     f"**{seam['dense_val_before']:.4f}** (dense, end of its run) to "
                     f"**{seam['moe_val_after']:.4f}** (MoE, before its first training step).")
    if "moe" in runs:
        m = runs["moe"]
        lines.append(f"- **The MoE keeps training:** over its {m['tokens_trained']:,} tokens, "
                     f"training loss went from {m['loss_curve'][0]:.4f} to "
                     f"**{m['final_loss']:.4f}** and validation loss from "
                     f"{m['initial_val_loss']:.4f} to **{m['final_val_loss']:.4f}**.")
    if "moe" in runs and "dense_continued" in runs:
        m, c = runs["moe"], runs["dense_continued"]
        diff = c["final_val_loss"] - m["final_val_loss"]
        verdict = ("lower than" if diff > 0 else "higher than" if diff < 0 else "equal to")
        lines.append(f"- **Fair comparison (same starting weights, same {m['tokens_trained']:,} "
                     f"tokens, same batches):** the MoE's final validation loss "
                     f"(**{m['final_val_loss']:.4f}**) is {verdict} the dense control's "
                     f"(**{c['final_val_loss']:.4f}**), a difference of **{abs(diff):.4f}**.")
    gaps, rising = [], []
    for name, r in runs.items():
        best_val = min(p["val_loss"] for p in r["val_curve"])
        gaps.append(f"`{name}` {r['final_val_loss'] - r['final_loss']:+.4f}")
        if r["final_val_loss"] > best_val:
            rising.append(f"`{name}` ends {r['final_val_loss'] - best_val:.4f} above its best")
    trend = ("validation loss is still at its lowest point at the end of every run, so none of "
             "them has started to overfit" if not rising else "; ".join(rising))
    lines.append("- **Overfitting check:** gap between final validation and training loss — "
                 + ", ".join(gaps) + f". {trend[0].upper() + trend[1:]}.")
    lines.append("")
    lines.append("![loss curves](results/loss_curves.png)")
    if os.path.exists(os.path.join(HERE, "results", "expert_balance.png")):
        lines.append("")
        lines.append("![expert balance](results/expert_balance.png)")
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
