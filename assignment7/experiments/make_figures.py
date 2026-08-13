"""Render every figure the README references, from the results JSON only.

Nothing here recomputes anything.  If a results file is missing, the figure is
skipped with a note rather than invented.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import FIGURES, read_json, rule  # noqa: E402

INK = "#1b1b1f"
KAS = "#c2410c"
DENSE = "#1d4ed8"
MUTED = "#8b8b93"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_e1() -> None:
    d = read_json("e1_invertibility.json")
    if not d:
        return print("  skip e1 (no results)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    names = list(d["gpt2_vocab"].keys())
    lost = [d["gpt2_vocab"][n]["n"] - d["gpt2_vocab"][n]["exact"] for n in names]
    colors = [DENSE if n.startswith("V1") else KAS for n in names]
    ax1.bar(range(len(names)), lost, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace(", ", "\n") for n in names], fontsize=7)
    ax1.set_ylabel("GPT-2 tokens NOT recovered")
    ax1.set_title("E1  round-trip failures on 50,257 tokens", loc="left", fontsize=9)
    for i, v in enumerate(lost):
        ax1.text(i, v + 1, str(v), ha="center", fontsize=8)

    ov = d["overflow"]
    xs = sorted(int(k) for k in ov)
    ax2.plot(xs, [ov[str(x)]["truncate"] for x in xs], "o-", color=KAS, label="truncate")
    ax2.plot(xs, [ov[str(x)]["alias"] for x in xs], "s--", color=MUTED, label="alias")
    ax2.set_xlabel("token length (bytes), L_max = 64")
    ax2.set_ylabel("byte recovery accuracy")
    ax2.set_title("E1  the design claim that lost", loc="left", fontsize=9)
    ax2.legend(frameon=False)
    save(fig, "e1_invertibility.png")


def fig_e3() -> None:
    d = read_json("e3_cost.json")
    if not d:
        return print("  skip e3 (no results)")
    p = d["params"]
    vs = sorted(int(k) for k in p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.plot(vs, [p[str(v)]["dense_head_params"] / 1e6 for v in vs], "o-",
             color=DENSE, label="dense head")
    ax1.plot(vs, [p[str(v)]["kas_head_params"] / 1e6 for v in vs], "s-",
             color=KAS, label="KAS head")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("vocabulary size"); ax1.set_ylabel("head parameters (M)")
    ax1.set_title("E3  the head stops scaling with |V|", loc="left", fontsize=9)
    ax1.legend(frameon=False)

    t = d.get("timing", {})
    # OOM entries carry no speedup; drop them rather than plotting a gap as zero.
    cpu = sorted(
        (t[k] for k in t if k.startswith("cpu") and t[k].get("speedup")),
        key=lambda r: r["V"],
    )
    if cpu:
        ax2.plot([r["V"] for r in cpu], [r["speedup"] for r in cpu],
                 "o-", color=KAS, label=f"CPU, N={cpu[0]['N']}")
    cu = sorted(
        (t[k] for k in t
         if k.startswith("cuda") and t[k]["N"] == 256 and t[k].get("speedup")),
        key=lambda r: r["V"],
    )
    if cu:
        ax2.plot([r["V"] for r in cu], [r["speedup"] for r in cu],
                 "s--", color=DENSE, label="CUDA, N=256")
    ax2.axhline(1.0, color=MUTED, lw=0.8)
    ax2.set_xscale("log"); ax2.set_xlabel("vocabulary size")
    ax2.set_ylabel("KAS speedup over dense (x)")
    ax2.set_title("E3  wall-clock, honestly", loc="left", fontsize=9)
    ax2.legend(frameon=False)
    save(fig, "e3_cost.png")


def fig_e4() -> None:
    d = read_json("e4_train_arms.json")
    if not d:
        return print("  skip e4 (no results)")
    arms = list(d["arms"].keys())
    bpb = [d["arms"][a]["val"]["bpb"] for a in arms]
    params = [d["arms"][a]["params"]["total"] / 1e6 for a in arms]
    floor = d["unigram_floor"]["bpb"]
    colors = [DENSE if a in ("a0_bpe_tied", "a1_v1_dense") else KAS for a in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax1.bar(range(len(arms)), bpb, color=colors)
    ax1.axhline(floor, color=MUTED, ls="--", lw=1)
    ax1.text(len(arms) - 0.4, floor, " unigram floor", va="bottom", ha="right",
             fontsize=7, color=MUTED)
    ax1.set_xticks(range(len(arms)))
    ax1.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=6.5)
    ax1.set_ylabel("validation bits per byte")
    ax1.set_ylim(min(bpb + [floor]) - 0.1, max(bpb + [floor]) + 0.1)
    ax1.set_title("E4  quality", loc="left", fontsize=9)

    ax2.scatter(params, bpb, c=colors, s=60, zorder=3)
    for a, x, y in zip(arms, params, bpb):
        ax2.annotate(a.replace("_", " "), (x, y), fontsize=6.5,
                     xytext=(4, 4), textcoords="offset points")
    ax2.axhline(floor, color=MUTED, ls="--", lw=1)
    ax2.set_xlabel("total parameters (M)")
    ax2.set_ylabel("validation bits per byte")
    ax2.set_title("E4  quality against size", loc="left", fontsize=9)
    save(fig, "e4_arms.png")


def fig_e5() -> None:
    d = read_json("e5_d_sweep.json")
    if not d or not d.get("runs"):
        return print("  skip e5 (no results)")
    runs = list(d["runs"].values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.scatter([r["max_coherence"] for r in runs], [r["bpb"] for r in runs],
                c=KAS, s=60, zorder=3)
    for r in runs:
        ax1.annotate(f"{r['d_c']}/{r['L_max']}", (r["max_coherence"], r["bpb"]),
                     fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel("byte-codebook max coherence")
    ax1.set_ylabel("validation bits per byte")
    ax1.set_title("E5  coherence predicts quality", loc="left", fontsize=9)

    ax2.scatter([r["D"] for r in runs], [r["bpb"] for r in runs], c=DENSE, s=60, zorder=3)
    for r in runs:
        ax2.annotate(f"{r['d_c']}/{r['L_max']}", (r["D"], r["bpb"]),
                     fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("D = d_c x L_max")
    ax2.set_ylabel("validation bits per byte")
    ax2.set_title("E5  D does not", loc="left", fontsize=9)
    save(fig, "e5_capacity.png")


def fig_e7() -> None:
    d = read_json("e7_ood_multiscript.json")
    if not d:
        return print("  skip e7 (no results)")
    langs = list(d["corpus"].keys())
    arms = list(d["arms"].keys())
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        vals = [d["arms"][arm][l] for l in langs]
        color = DENSE if arm in ("a0_bpe_tied", "a1_v1_dense") else KAS
        ax.bar([x + i * w for x in range(len(langs))], vals, width=w,
               label=arm, color=color, alpha=0.55 + 0.45 * (i % 2))
    ax.set_xticks([x + 0.4 - w / 2 for x in range(len(langs))])
    ax.set_xticklabels([f"{l}\n{d['corpus'][l]['label']}" for l in langs], fontsize=7)
    ax.set_ylabel("bits per byte")
    ax.set_title("E7  out-of-domain, by script", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "e7_ood.png")


def fig_e8() -> None:
    d = read_json("e8_token_buckets.json")
    if not d:
        return print("  skip e8 (no results)")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for arm, rec in d["arms"].items():
        ax.plot(range(len(rec["delta_nll_by_decile"])), rec["delta_nll_by_decile"],
                "o-", label=arm)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xlabel("validation frequency decile (0 = most frequent)")
    ax.set_ylabel(f"delta NLL vs {d['baseline']} (nats/token)")
    ax.set_title("E8  where the constraint helps and hurts", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "e8_buckets.png")


def main() -> None:
    rule("rendering figures")
    for fn in (fig_e1, fig_e3, fig_e4, fig_e5, fig_e7, fig_e8):
        fn()


if __name__ == "__main__":
    main()
