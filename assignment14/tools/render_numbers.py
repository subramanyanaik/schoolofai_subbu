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
        "| run | mode | total params | active params | steps | tokens trained | "
        "final loss | min loss | tokens/s | wall clock | peak memory |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    totals = {"dense": pc.get("dense_total"), "moe": pc.get("moe_total")}
    actives = {"dense": pc.get("dense_total"), "moe": pc.get("moe_active")}
    for name, r in runs.items():
        mem = f"{r['peak_memory_mib']:.0f} MiB" if r.get("peak_memory_mib") is not None else "n/a"
        total = totals.get(name)
        active = actives.get(name)
        lines.append(
            f"| `{name}` | {r['mode']} | {total:,} | {active:,} | {r['steps']:,} | "
            f"{r['tokens_trained']:,} | {r['final_loss']:.4f} | {r['min_loss']:.4f} | "
            f"{r['tokens_per_s']:,.0f} | {r['wall_clock_s'] / 60:.1f} min | {mem} |")
    lines.append("")

    if pc:
        ratio_total = pc["moe_total"] / pc["dense_total"]
        ratio_active = pc["moe_active"] / pc["dense_total"]
        lines.append(f"- MoE total parameters are **{ratio_total:.2f}x** the dense model's; "
                     f"MoE *active* parameters per token are **{ratio_active:.2f}x** the dense "
                     f"model's (**{pc['moe_active'] / pc['moe_total']:.1%}** of the MoE total).")
    seam = results.get("seam_loss_right_after_conversion")
    if seam is not None and "dense" in runs:
        lines.append(f"- loss immediately after conversion, before any MoE-phase training: "
                      f"**{seam:.4f}** (dense model's own final training loss: "
                      f"**{runs['dense']['final_loss']:.4f}**).")
    if "moe" in runs and "dense" in runs:
        d_final, m_final = runs["dense"]["final_loss"], runs["moe"]["final_loss"]
        lines.append(f"- after continued training, the MoE model's final loss "
                      f"(**{m_final:.4f}**) is {'below' if m_final < d_final else 'above'} the "
                      f"dense model's final loss (**{d_final:.4f}**).")
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
