"""Render the write-up's numbers from results/results.json, and check the README matches.

(Named render_numbers rather than numbers because `tools/` lands on sys.path[0] when these
scripts are run directly, and a module called `numbers` there shadows the stdlib one that
`decimal` -- and therefore half of jsonschema -- imports.)

The README must not be able to drift away from the run that produced it, so the tables of
numbers live between markers and are generated from the JSON the notebook emits.

    python tools/render_numbers.py --write    # inject the current numbers into README.md
    python tools/render_numbers.py --check    # exit 1 if README.md disagrees with results.json
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results", "results.json")
README = os.path.join(HERE, "README.md")
BEGIN, END = "<!-- BEGIN:numbers -->", "<!-- END:numbers -->"


def render(r):
    L = []
    a = L.append
    a("### Part 1 — the seven numbers")
    a("")
    a("| # | what was measured | the number |")
    a("|---|---|---|")
    a(f"| **1** | shapes, and the size of the last axis | `hidden [B,T,D]` holds "
      f"**{r['item1_hidden_numel']:,}** numbers, `logits [B,T,V]` holds "
      f"**{r['item1_logits_numel']:,}** — a **{r['item1_blowup']}×** blow-up (V/D), "
      f"created in the last layer |")
    a(f"| **2** | the shift, read as strings | target stream **==** input stream advanced by "
      f"one, asserted *and* printed as text; the wrong-direction table is printed beside it |")
    a(f"| **3** | contributing tokens, padding counted → masked | "
      f"**{r['item3_tokens_counted']:,} → {r['item3_tokens_masked']:,}** "
      f"({r['item3_pad_fraction_pct']}% of the naive `B*(T-1)` denominator was padding) |")
    a(f"| **4** | loss before → after masking one document boundary | "
      f"**{r['item4_loss_before']:.4f} → {r['item4_loss_after']:.4f}** nats "
      f"({r['item4_n_before']} → {r['item4_n_after']} sites); that one site's own term is "
      f"**{r['item4_boundary_term']:.4f}** nats. Over **{r['item4_pairs']}** independently "
      f"packed pairs the boundary averages **{r['item4_mean_boundary']:.4f}** against "
      f"**{r['item4_mean_interior']:.4f}** everywhere else — **{r['item4_mean_ratio']}×** — "
      f"and is the worse site in **{r['item4_boundary_worse_pct']:.0f}%** of them |")
    a(f"| **5** | perplexity of an untrained model vs V | "
      f"**{r['item5_init_ppl']:,.1f}** against **V = {r['vocab_size']:,}** "
      f"(ratio **{r['item5_ppl_over_vocab']}**); loss {r['item5_init_loss']:.4f} vs "
      f"ln V = {r['ln_vocab']:.4f} |")
    a(f"| **6** | tied vs untied head parameters | "
      f"**{r['item6_untied_total']:,} → {r['item6_tied_total']:,}**, saving "
      f"**{r['item6_saved']:,}** ({r['item6_saved_pct']}% of the model) |")
    a(f"| **7** | peak memory, ordinary vs chunked cross-entropy | "
      f"**{r['item7_peak_naive_mib']:,.1f} MiB → {r['item7_peak_chunked_mib']:,.1f} MiB "
      f"= {r['item7_ratio']}×**, same loss to {r['item7_loss_absdiff']:.0e}, same gradients "
      f"to {r['item7_grad_absdiff']:.0e} |")
    a("")
    sweep = r.get("item7_sweep")
    if sweep:
        a(f"Item 7's chunk-size sweep, `N = {r['item7_N']:,}` tokens against "
          f"`V = {r['vocab_size']:,}`, ordinary cross-entropy at "
          f"**{r['item7_peak_naive_mib']:,.1f} MiB**:")
        a("")
        a("| chunk | " + " | ".join(str(s["chunk"]) for s in sweep) + " |")
        a("|---|" + "---|" * len(sweep))
        a("| peak (MiB) | " + " | ".join(f"{s['peak_mib']:,.1f}" for s in sweep) + " |")
        a("| vs ordinary | " + " | ".join(f"{s['ratio']}×" for s in sweep) + " |")
        a("| loss | " + " | ".join(f"{s['loss']:.6f}" for s in sweep) + " |")
        a("")
    a("### Part 2 — the two losses")
    a("")
    a("| head | predicts | loss (nats) | perplexity |")
    a("|---|---|---|---|")
    a(f"| head 1 | `t+1` | **{r['part2_head1_loss']:.4f}** | {r['part2_head1_ppl']:,.1f} |")
    a(f"| head 2 | `t+2` | **{r['part2_head2_loss']:.4f}** | {r['part2_head2_ppl']:,.1f} |")
    a(f"| **sum** | the quantity actually optimised | **{r['part2_sum']:.4f}** | — |")
    a(f"| gap | head 2 − head 1 | **{r['part2_gap']:.4f}** | — |")
    a("")
    a(f"Single-head baseline trained on identical batches with an identical seed: "
      f"**{r['part2_baseline_head1_loss']:.4f}** nats, so the extra head moved head 1 by "
      f"**{r['part2_head1_delta_vs_baseline']:+.4f}** nats.")
    a("")
    a("### Part 3 — the warning, demonstrated")
    a("")
    a(f"| | training loss (nats) | perplexity |")
    a("|---|---|---|")
    a(f"| correct shift, {r['part2_steps']} steps | {r['part3_correct_final_loss']:.4f} | "
      f"{__import__('math').exp(r['part3_correct_final_loss']):,.1f} |")
    a(f"| **off-by-one shift, {r['part3_bug_steps']} steps** | "
      f"**{r['part3_bug_final_loss']:.4f}** | **{r['part3_bug_final_ppl']:.2f}** |")
    a("")
    a(f"…and **{r['part3_copy_rate_pct']}%** of that model's predictions are literal copies "
      f"of its own input.")
    a("")
    a(f"*Produced by `loss_harness.ipynb` on {r.get('gpu') or 'CPU'}, torch "
      f"{r['torch_version']}, seed {r['seed']}. Every figure above is read straight out of "
      f"[`results/results.json`](results/results.json) by `tools/render_numbers.py`. "
      f"GPU float reductions are not bit-reproducible, so a re-run moves the last decimals "
      f"and the JSON, the log and this table are always re-generated together.*")
    return "\n".join(L)


def main():
    r = json.load(io.open(RESULTS, encoding="utf-8"))
    block = render(r)
    if "--print" in sys.argv:
        print(block)
        return 0
    text = io.open(README, encoding="utf-8").read()
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < 0:
        print(f"README.md is missing the {BEGIN} / {END} markers")
        return 1
    current = text[i + len(BEGIN):j].strip()
    if "--check" in sys.argv:
        if current != block.strip():
            print("README.md numbers disagree with results/results.json.")
            print("Run: python tools/render_numbers.py --write")
            return 1
        print("README.md matches results/results.json")
        return 0
    io.open(README, "w", encoding="utf-8", newline="\n").write(
        text[:i + len(BEGIN)] + "\n\n" + block.strip() + "\n\n" + text[j:])
    print("README.md numbers updated from results/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
