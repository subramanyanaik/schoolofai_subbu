"""Render the write-up's numbers from results/results.json, and check the README matches.

(Named render_numbers rather than numbers because `tools/` lands on sys.path[0] when these
scripts are run directly, and a module called `numbers` there shadows the stdlib one that
`decimal` -- and therefore half of jsonschema -- imports.)

The README must not be able to drift away from the run that produced it, so everything
between the markers is generated from the JSON the notebook emits.

    python tools/render_numbers.py --write    # inject the current numbers into README.md
    python tools/render_numbers.py --check    # exit 1 if README.md disagrees with results.json
    python tools/render_numbers.py --print    # print the block without touching README.md
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

    # ------------------------------------------------------------------ 1. shapes
    a("## 1. Every tensor shape in the step")
    a("")
    a(f"One step at `B={r['trace_B']}`, `T={r['trace_T']}`, on a "
      f"{r['n_params']:,}-parameter model with `C={r['cfg_n_embd']}`, `L={r['cfg_n_layer']}` "
      f"blocks, `nh={r['cfg_n_head']}` heads and `V={r['cfg_vocab_size']:,}`. "
      f"**{r['n_traced_tensors']} tensors** are printed with the meaning of every axis; the "
      f"full trace with all {r['cfg_n_layer']} blocks expanded "
      f"({r['n_trace_rows_full']} rows) is in "
      f"[`results/shapes_full.txt`](results/shapes_full.txt).")
    a("")
    a("| what | value |")
    a("|---|---|")
    a(f"| activations traced in one step | **{r['act_mib']:,.1f} MiB** |")
    a(f"| of which `logits [B,T,V]` alone | **{r['logit_mib']:,.1f} MiB "
      f"({r['logit_share_pct']}%)**, because `V/C = {r['v_over_c']:.0f}` |")
    a(f"| weights + grads + Adam's two moments | **{r['state_mib']:,.1f} MiB** = "
      f"**{r['bytes_per_param']} bytes per parameter** at fp32 "
      f"({r['n_param_tensors']} parameters x 4 tensors each) |")
    a(f"| blocks are shape-identical | {r['n_interior_tensors']} interior tensors, the same "
      f"in all {r['cfg_n_layer']} |")
    a("")
    a("The three audits, because a shape table written from memory proves nothing:")
    a("")
    a("| audit | result |")
    a("|---|---|")
    a(f"| every leaf module's output shape appears in the trace | "
      f"**{r['audit_hooked_modules']} module calls**, 0 shapes missing |")
    a(f"| `dL/dX` has `X`'s own shape, for every traced activation | "
      f"**{r['audit_grad_shape_checked']} checked**, 0 mismatches — so the backward pass "
      f"introduces no shape the forward pass did not already show |")
    a(f"| the traced explicit attention == the fused kernel training uses | max logit "
      f"difference **{r['audit_fused_logit_maxdiff']:.1e}**, loss difference "
      f"**{r['audit_fused_loss_absdiff']:.1e}** |")
    a("")
    a(f"The last one matters for a reason worth stating: the `[B, nh, T, T]` attention-score "
      f"tensor printed in the trace is real, but the fused kernel the training loop actually "
      f"runs **never builds it**. That is what flash attention *is*. Printing every tensor "
      f"required a second, explicit code path, and then required proving the two agree.")
    a("")

    # ------------------------------------------------------------------ 2. gradient
    a("## 2. One gradient, verified by hand")
    a("")
    a(f"`{r['fd_param']}`, in float64 on the CPU, dropout off, one fixed batch.")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| `backward()` reports | `{r['fd_autograd']:.15f}` |")
    a(f"| central difference at the best `h` = {r['fd_best_h']:.0e} | "
      f"`{r['fd_best_central']:.15f}` |")
    a(f"| relative error | **{r['fd_best_rel']:.2e}** |")
    a(f"| **they agree to** | **{r['fd_best_digits']} significant decimal digits** |")
    a("")
    a("The step size sweep is the point of the exercise, not a detail — there is a best `h` "
      "and the error rises on *both* sides of it:")
    a("")
    sw = r["fd_sweep"]
    a("| h | " + " | ".join(f"{s['h']:.0e}" for s in sw) + " |")
    a("|---|" + "---|" * len(sw))
    a("| digits agreeing | " + " | ".join(f"{s['digits']:.1f}" for s in sw) + " |")
    a("")
    a(f"Right of the minimum the estimator measures a chord instead of a tangent; left of it "
      f"`L(w+h) - L(w-h)` is a subtraction of two numbers that agree in their leading digits "
      f"and there is nothing left to divide. Same check in float32: **only "
      f"{r['fd32_best_digits']} digits** confirmable, at best. That is a fact about the "
      f"ruler — the fp32 and fp64 gradients themselves agree to "
      f"{r['fd32_vs_fd64_digits']} digits.")
    a("")
    a("| broadened check | result |")
    a("|---|---|")
    a(f"| ten more parameters, one from each kind in the model | worst agreement "
      f"**{r['fd_multi_worst_digits']} digits** (`{r['fd_multi_worst_name']}`) |")
    a(f"| directional derivative over all {r['n_params']:,} parameters at once | measured "
      f"`{r['dirderiv_fd']:.12f}` vs reported `g·u` = `{r['dirderiv_dot']:.12f}` — "
      f"**{r['dirderiv_digits']} digits** |")
    a("")
    a(f"**And when it fails.** Turn dropout on, change nothing else: the same loss evaluated "
      f"four times at the same weights spreads by **{r['noisy_spread']:.2e}**, while the "
      f"nudge only has to move the loss by ~**{r['noisy_signal']:.0e}**. The check now "
      f"reports {r['noisy_digits']} digits of agreement — the finite difference and "
      f"`backward()` no longer share a single significant digit. `backward()` is still "
      f"exactly right; `L(w)` has stopped being a function. The diagnostic is one line — "
      f"evaluate "
      f"the loss twice without touching anything, and if those two numbers differ, the "
      f"finite-difference check cannot be run yet.")
    a("")
    a(f"**The bug found on the way in.** "
      f"{'Confirmed in this run: ' if r.get('sdpa_fp64_cpu_nan') else ''}"
      f"`F.scaled_dot_product_attention` returns **NaN** for CPU float64 inputs with "
      f"`dropout_p=0` on torch {r['torch_version']} — in the forward pass, before any "
      f"differentiation. With `dropout_p=0.1` a different kernel is chosen and the NaN "
      f"disappears, which is the worst possible symptom. Minimal repro in the notebook.")
    a("")

    # ------------------------------------------------------------------ 3. accumulation
    a("## 3. Gradient accumulation, broken on purpose")
    a("")
    a(f"The lengths are real: Tiny Shakespeare split at blank lines gives **{r['n_docs']:,} "
      f"documents**, median **{r['doc_len_median']} tokens**, longest "
      f"**{r['doc_len_max']}**. Short documents are a different register from long ones "
      f"(speaker headings and one-line exchanges versus speeches), so mis-weighting them is "
      f"a *bias*, not extra noise. One step, four micro-batches — two drawn from the long "
      f"pool, two from the short:")
    a("")
    ns, wc, wb = r["accum_ns"], r["accum_w_correct"], r["accum_w_broken"]
    a("| micro-batch | " + " | ".join(str(i) for i in range(len(ns))) + " | total |")
    a("|---|" + "---|" * (len(ns) + 1))
    a("| contributing tokens `n_i` | " + " | ".join(f"{n:,}" for n in ns) +
      f" | {r['accum_N']:,} |")
    a("| correct weight `n_i/N` | " + " | ".join(f"{w:.4f}" for w in wc) + " | 1.0000 |")
    a("| broken weight `1/G` | " + " | ".join(f"{w:.4f}" for w in wb) + " | 1.0000 |")
    a("| over/under-weighted | " + " | ".join(f"{b / c:.2f}×" for b, c in zip(wb, wc)) +
      " | — |")
    a("")
    a(f"Both weight columns sum to 1, so this is **not** a learning-rate bug in disguise. "
      f"Short documents are weighted **{r['accum_max_overweight']}× too heavily**, long ones "
      f"**{r['accum_min_underweight']}× too lightly**.")
    a("")
    a("| the two gradients, same weights, same four micro-batches | |")
    a("|---|---|")
    a(f"| cosine similarity | **{r['accum_cos']:.6f}** ({r['accum_angle_deg']}° apart) |")
    a(f"| ‖broken‖ / ‖correct‖ | **{r['accum_norm_ratio']}** — "
      f"{abs(1 - r['accum_norm_ratio']) * 100:.0f}% apart in magnitude against "
      f"{r['accum_angle_deg']:.0f}° apart in direction, so no learning-rate change repairs "
      f"it |")
    a(f"| ‖broken − correct‖ / ‖correct‖ | **{r['accum_rel_l2']}** |")
    a("")
    a("And the definition that settles which one is *correct*, rather than merely different "
      "— put every row through in a single forward pass and compare:")
    a("")
    a("| | max abs difference from the un-accumulated gradient |")
    a("|---|---|")
    a(f"| correct rule (token-weighted) | **{r['accum_oneshot_maxdiff_correct']:.1e}** — the "
      f"same number |")
    a(f"| broken rule (average of averages) | {r['accum_oneshot_maxdiff_broken']:.1e} |")
    a("")
    a(f"**Why nobody noticed until 2024.** Repeat with equal-length micro-batches and the two "
      f"rules agree to cosine **{r['accum_equal_cos']:.12f}**, max difference "
      f"**{r['accum_equal_maxdiff']:.1e}**. Under fixed-length packing the bug does not "
      f"exist. It needs masked padding, masked prompts or ragged documents to appear — which "
      f"is exactly what instruction tuning introduced.")
    a("")
    a(f"Two {r['accum_steps']}-step runs, identical initialisation, identical data stream, "
      f"identical seeds, differing only in that scalar, then judged by the *same* ruler — the "
      f"correctly token-weighted loss on held-out documents:")
    a("")
    a("| held-out loss (nats/token) | correct rule | broken rule | gap |")
    a("|---|---|---|---|")
    a(f"| all documents | **{r['accum_final_correct']:.4f}** | "
      f"**{r['accum_final_broken']:.4f}** | **{r['accum_final_gap']:+.4f}** |")
    a(f"| short documents | {r['accum_final_correct_short']:.4f} | "
      f"{r['accum_final_broken_short']:.4f} | "
      f"{r['accum_final_broken_short'] - r['accum_final_correct_short']:+.4f} |")
    a(f"| long documents | {r['accum_final_correct_long']:.4f} | "
      f"{r['accum_final_broken_long']:.4f} | "
      f"{r['accum_final_broken_long'] - r['accum_final_correct_long']:+.4f} |")
    a("")
    a("![accumulation](results/accumulation.png)")
    a("")
    a("The third panel is the one that shows *where the weight went*: the broken run is "
      "worse on the documents it under-weighted and closer on the ones it over-weighted. "
      "The bug did not add noise, it optimised a different objective.")
    a("")

    # ------------------------------------------------------------------ 4. grad norm
    a("## 4. Grad norm at every step, and one step where it moved first")
    a("")
    a(f"Logged at every one of {r['log_steps']} steps, in four runs. Alongside the training "
      f"loss, a **fixed probe batch** is evaluated *before* each update, so \"before\" is "
      f"well defined: probe[k] is measured, then the gradient of batch k is taken, then the "
      f"update is applied. A spike in `gradnorm[k]` cannot reach `probe[k]`.")
    a("")
    a("**The general claim, tested first, and it does not survive.** The lag at which changes "
      "in grad norm correlate best with changes in the probe loss:")
    a("")
    a(f"| run | peak lag | correlation |")
    a("|---|---|---|")
    a(f"| healthy, `lr={r['log_lr']}` | **{r['lag_best_nat']}** | "
      f"{r['lag_best_nat_corr']} |")
    a(f"| aggressive, `lr={r['hot_lr']}` | **{r['lag_best_hot']}** | "
      f"{r['lag_best_hot_corr']} |")
    a("")
    lead_word = ("Neither peaks at a positive lag"
                 if max(r["lag_best_nat"], r["lag_best_hot"]) <= 0 else
                 "The peaks are barely off zero")
    a(f"{lead_word}, and both correlations are weak. On an ordinary step the grad norm and "
      f"the loss are both reporting the same batch, so of course they move together. The "
      f"lead is not a property of every step — it is a property of the steps where something "
      f"is going wrong, and those have to be found one at a time.")
    a("")
    a("**The search, at every threshold I might have picked** (number of qualifying steps — "
      "norm several σ out, loss quiet at that step, loss breaking trend within 8 steps):")
    a("")
    grid = r["lead_grid"]
    zgs = sorted({g["z_g"] for g in grid})
    a("| z(grad norm) > | " + " | ".join(f"{z:.0f}" for z in zgs) + " |")
    a("|---|" + "---|" * len(zgs))
    for za in sorted({g["z_after"] for g in grid}):
        row_n = [next(g["natural"] for g in grid if g["z_g"] == z and g["z_after"] == za)
                 for z in zgs]
        row_h = [next(g["hot"] for g in grid if g["z_g"] == z and g["z_after"] == za)
                 for z in zgs]
        a(f"| healthy run, loss later > {za:.0f}σ | " + " | ".join(str(v) for v in row_n) +
          " |")
        a(f"| aggressive run, loss later > {za:.0f}σ | " + " | ".join(str(v) for v in row_h) +
          " |")
    a("")
    if r.get("lead_event_step") is not None:
        a(f"**The one natural instance**, step {r['lead_event_step']} of the "
          f"{r['lead_event_source']} run: grad norm "
          f"**{r['lead_event_gnorm']:.4f}** against a trailing median of "
          f"{r['lead_event_gnorm_median']:.4f} (**{r['lead_event_z_gnorm']}σ**), while the "
          f"probe loss sat at {r['lead_event_probe']:.4f} against a median of "
          f"{r['lead_event_probe_median']:.4f} (**{r['lead_event_z_probe']:+}σ** — "
          f"unmoved). **{r['lead_event_lead']} steps later** the probe was "
          f"{r['lead_event_probe_after']:.4f} ({r['lead_event_z_after']}σ). Reported for "
          f"what it is: one marginal event in {2 * (r['log_steps'] - 50):,} logged steps, "
          f"with the probe moving by "
          f"{r['lead_event_probe_after'] - r['lead_event_probe']:+.4f} nats.")
    else:
        a("**No natural instance qualified** in either run at a threshold I would defend. "
          "Reported as found.")
    a("")
    a(f"**The unambiguous one**, where the cause is supplied rather than waited for: a "
      f"corrupted shard at step {r['inject_at']} — three steps in which 25% of the target "
      f"ids are replaced with random ones, the sort of thing a bad data pipeline actually "
      f"does.")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| grad norm before | {r['inject_gnorm_median_before']} (trailing median) |")
    a(f"| grad norm at the shard | **{r['inject_gnorm_peak']}** — "
      f"**{r['inject_gnorm_ratio']}×** |")
    a(f"| probe loss at that same step | {r['inject_probe_at_spike']:.4f} against a median "
      f"of {r['inject_probe_median_before']:.4f} — **{r['inject_probe_sigma_at_spike']:+}σ, "
      f"i.e. exactly on trend** |")
    a(f"| probe loss first breaks trend | **{r['inject_lead']} steps later** |")
    a(f"| same shard, same seed, with `clip_grad_norm_(1.0)` | final probe "
      f"{r['inject_probe_final_clipped']:.4f} vs {r['inject_probe_final_noclip']:.4f} "
      f"unclipped |")
    a("")
    a("![gradnorm](results/gradnorm.png)")
    a("")
    a("That is the whole case for logging the norm: at the step where the damage entered, "
      "the loss curve showed nothing at all.")
    a("")

    # ------------------------------------------------------------------ 5. MFU
    a("## 5. MFU, computed honestly")
    a("")
    a(f"**The denominator is not a spec sheet.** This is a laptop GPU whose clocks move with "
      f"temperature, and quoting a marketing number would make the percentage "
      f"unfalsifiable. The ceiling was measured in the same process, minutes before the "
      f"training run:")
    a("")
    a("| measured ceiling on this machine | TFLOP/s |")
    a("|---|---|")
    a(f"| fp32, TF32 disabled | {r['peak_fp32_tflops']} |")
    a(f"| fp32 with TF32 | {r['peak_tf32_tflops']} |")
    a(f"| bf16 tensor cores | **{r['peak_bf16_tflops']}** "
      f"({r['bf16_over_fp32']}× the fp32 pipeline) |")
    a("")
    a(f"Taken from the 4096-cube matmul rather than the fastest of the sizes tried: a "
      f"1024-cube is 2 GFLOP and fits in cache, so quoting it would inflate the denominator "
      f"and *deflate* every MFU below. And the honest caveat on the third digit — the worst "
      f"spread between repeat timings of the *same* matmul on this laptop was "
      f"**{r['peak_spread_pct']}%**, so every percentage in this section carries at least "
      f"that much uncertainty.")
    a("")
    a(f"**The numerator** is nanoGPT's estimate, `6N + 12·L·nh·hs·T` = "
      f"**{r['flops_per_token'] / 1e6:.2f} MFLOP/token** at `T={r['cfg_block_size']}`, of "
      f"which the output head is **{r['flops_head_pct']}%** and the attention term "
      f"**{r['flops_attn_pct']}%**.")
    a("")
    ph = r["mfu_base_phases"]
    a(f"**The baseline step** — {r['ladder'][0]['label']} — takes "
      f"**{r['mfu_base_step_ms']} ms**: forward {ph['fwd']} ms, backward {ph['bwd']} ms, "
      f"optimizer + clipping {ph['opt']} ms, everything else {ph['other']} ms. That is "
      f"{r['mfu_base_tokens_per_s']:,.0f} tokens/s = {r['mfu_base_tflops']} TFLOP/s = "
      f"**{r['mfu_base_pct_fp32']}%** of the fp32 ceiling it runs in, or "
      f"**{r['mfu_base_pct_bf16']}%** of the bf16 ceiling everyone quotes.")
    a("")
    a("**One change at a time:**")
    a("")
    a("| configuration | tokens/s | TFLOP/s | MFU vs measured bf16 peak |")
    a("|---|---|---|---|")
    for row in r["ladder"]:
        if row.get("oom"):
            a(f"| {row['label']} | — | — | out of memory on a 4 GiB card |")
        else:
            a(f"| {row['label']} | {row['tokens_per_s']:,.0f} | {row['tflops']:.3f} | "
              f"**{row['mfu']:.1f}%** |")
    if not r.get("compile_ok"):
        a(f"| + `torch.compile` | — | — | unavailable here: "
          f"`{r.get('compile_error', '')[:60]}` |")
    a("")
    a("![mfu](results/mfu.png)")
    a("")
    a(f"**The honest account of the distance to 40%.** Rather than list suspicions, benchmark "
      f"the model's *actual* matmul shapes in bf16 and ask what a step made of nothing but "
      f"those matmuls — no LayerNorm, no softmax, no optimizer, no launch overhead, no "
      f"Python — could possibly achieve. That is an upper bound the loop cannot beat:")
    a("")
    a("| | MFU |")
    a("|---|---|")
    a(f"| best measured | **{r['mfu_best_pct']}%** ({r['mfu_best_label']}) |")
    a(f"| ceiling imposed by the matmul shapes alone | **{r['shape_ceiling_mfu']}%** "
      f"({r['shape_ceiling_tflops']} TFLOP/s) |")
    a(f"| the target | 40% |")
    a("")
    a(f"That splits the distance cleanly:")
    a("")
    a(f"* **100% → {r['shape_ceiling_mfu']}% ({r['mfu_shape_cost']} points) is shape**, gone "
      f"before the loop starts. The output head is {r['head_share_of_matmul_flops']}% of the "
      f"arithmetic and is one `[B·T, {r['cfg_n_embd']}] × [{r['cfg_n_embd']}, "
      f"{r['cfg_vocab_size']:,}]` matmul; with `K={r['cfg_n_embd']}` a tensor core spends "
      f"most of its time reading rather than multiplying, and it reaches "
      f"**{r['head_pct_of_square_bf16']}%** of the square-matmul rate "
      f"({r['square_bf16_tflops']} TFLOP/s). The worst-shaped matmul in the model is "
      f"`attn c_proj` at {r['worst_shape_pct_sq']}%, but it is only "
      f"{r['worst_shape_flops_pct']}% of the work. A wider model fixes this; a better loop "
      f"does not.")
    a(f"* **{r['shape_ceiling_mfu']}% → {r['mfu_best_pct']}% ({r['mfu_loop_cost']} points) "
      f"is everything that is not a matmul** — the optimizer and gradient clipping "
      f"({r['opt_share_pct']}% of the step, pure memory traffic over "
      f"{r['n_params'] / 1e6:.0f}M parameters × 4 tensors, contributing nothing to the "
      f"numerator); LayerNorm, GELU, softmax and the residual adds, all memory-bound; and "
      f"the fixed per-step launch and Python overhead, which is why the batch-size sweep "
      f"runs {r['mfu_sweep_min']}% at batch 1 and {r['mfu_sweep_max']}% at batch "
      f"{r['mfu_sweep_best_batch']} (a separate sweep from the ladder above, so not the same "
      f"number to the decimal)"
      + ("; and no kernel fusion, since `torch.compile` does not run in this environment"
         if not r.get("compile_ok") else "") + ".")
    ok_sweep = [s for s in r["mfu_sweep"] if not s.get("oom")]
    if ok_sweep and ok_sweep[-1]["mfu"] < r["mfu_sweep_max"]:
        a("")
        a(f"The sweep also turns over: MFU climbs to {r['mfu_sweep_max']}% at batch "
          f"{r['mfu_sweep_best_batch']} and then falls back to "
          f"{ok_sweep[-1]['mfu']:.1f}% at batch {ok_sweep[-1]['batch']}. Nothing about the "
          f"arithmetic changed — the card ran out of headroom and the allocator started "
          f"working for a living. On 4 GiB, \"use a bigger batch\" has an end.")
    a("")
    if r["shape_ceiling_mfu"] >= 40:
        a(f"**So the answer is the opposite of the one I expected.** 40% is not out of reach "
          f"at this model shape — the shapes allow {r['shape_ceiling_mfu']}%. The loop "
          f"captures **{r['mfu_capture_pct']}%** of what they allow; to hit 40% it would "
          f"have to capture {r['mfu_capture_needed_pct']}%. On this machine the distance to "
          f"40% is a loop-and-batch-size problem, not a model-shape problem. Had the head "
          f"been narrower still, or the batch smaller, the verdict would have gone the other "
          f"way — which is exactly why it is worth measuring rather than asserting.")
    else:
        a(f"**So 40% is not reachable at this model shape at all**: the matmuls cap it at "
          f"{r['shape_ceiling_mfu']}%, and no change to the training loop moves that.")
    a("")
    a(f"What I am **not** claiming is that the bullets under the second point sum to "
      f"{r['mfu_loop_cost']} points — they are measured contributions, not a partition, and "
      f"some of the gap is unaccounted for. The largest lever I actually had was precision "
      f"and batch size together, worth **{r['ladder_gain']} points**; the largest one I did "
      f"not have is making the model wider.")
    a("")

    # ------------------------------------------------------------------ 6. formats
    a("## 6. The number 0.1 in fp32, bf16 and fp8 E4M3")
    a("")
    a("0.1 has no finite binary expansion: `0.1 = 0.0 0011 0011 0011…₂ = "
      "1.10011001100…₂ × 2⁻⁴`, the block `0011` repeating for ever, exactly as `1/3` repeats "
      "in base 10. **No binary format of any width stores 0.1.** Each one stores some other "
      "number and calls it 0.1; the only question is which.")
    a("")
    a("Derived by hand in exact rational arithmetic from `1/10`, with round-to-nearest-"
      "ties-to-even, then checked against the bits torch actually stores:")
    a("")
    a("| format | sign · exponent · mantissa | value stored | relative error | matches torch |")
    a("|---|---|---|---|---|")
    b32, bb16, b16, b8 = (r["bits_fp32"], r["bits_bf16"], r["bits_fp16"], r["bits_fp8_e4m3"])
    a(f"| **fp32** (1+8+23) | `{b32[0]}` `{b32[1:9]}` `{b32[9:]}` | "
      f"0.100000001490116119384765625 | {r['fp32_rel_err_0p1']:.2e} | ✓ |")
    a(f"| **bf16** (1+8+7) | `{bb16[0]}` `{bb16[1:9]}` `{bb16[9:]}` | 0.10009765625 | "
      f"{r['bf16_rel_err_0p1']:.4f}% | ✓ |")
    a(f"| fp16 (1+5+10) | `{b16[0]}` `{b16[1:6]}` `{b16[6:]}` | 0.0999755859375 | "
      f"0.0244% | ✓ |")
    a(f"| **fp8 E4M3** (1+4+3) | `{b8[0]}` `{b8[1:5]}` `{b8[5:]}` | 0.1015625 | "
      f"**{r['fp8_rel_err_0p1']:.4f}%** | ✓ |")
    a("")
    a(f"Worked through for fp8 E4M3, since it is the one that is easy to get wrong: exponent "
      f"field `0011` = 3, bias 7, so `2^(3-7) = 2⁻⁴`; mantissa field `101` = `1 + 5/8` = "
      f"1.625; `1.625 × 2⁻⁴ = 0.1015625`. The true mantissa 1.6 sits between 1.5 (`100`) and "
      f"1.625 (`101`) and rounds to the nearer, which is 1.625. Three mantissa bits is "
      f"**{r['fp8_rel_err_0p1']:.2f}% error on a number as ordinary as 0.1**.")
    a("")
    a(f"The same encoder, run against torch over **10,000 values** including subnormals and "
      f"the saturation edge: **{r['encoder_mismatch_bf16']} bf16 mismatches, "
      f"{r['encoder_mismatch_fp8']} fp8 mismatches**. The hand method is the machine's "
      f"method.")
    a("")
    a("| format | exponent | mantissa | decimal digits | min subnormal | min normal | max |")
    a("|---|---|---|---|---|---|---|")
    for p in r["format_props"]:
        a(f"| {p['name']} | {p['e_bits']} | {p['m_bits']} | {p['decimal_digits']:.1f} | "
          f"{p['min_subnormal']:.2e} | {p['min_normal']:.2e} | {p['max_val']:,.4g} |")
    a("")
    a("bf16 and fp32 have the **same 8 exponent bits** — identical range, 2¹⁶ times coarser "
      "steps. fp16 spends 3 of those exponent bits on mantissa instead, and that trade is "
      "the entire reason loss scaling exists.")
    a("")
    a("### Which one would I train in")
    a("")
    a("**bf16 for the compute; fp32 for the master weights, the optimizer state and the "
      "loss.** The reason is not the table above — it is this model's own gradients. One "
      "real backward pass, then every format applied to the tensor it produced:")
    a("")
    a("| | share of gradient entries below it |")
    a("|---|---|")
    a(f"| fp16 min normal `2⁻¹⁴` | **{r['grad_pct_below_fp16_normal']}%** |")
    a(f"| fp16 min subnormal `2⁻²⁴` | {r['grad_pct_below_fp16_sub']}% |")
    a(f"| fp8 E4M3 min normal `2⁻⁶` | **{r['grad_pct_below_e4m3_normal']}%** |")
    a(f"| bf16 min normal `2⁻¹²⁶` | **{r['grad_pct_below_bf16_normal']}%** |")
    a("")
    a("| gradient cast to | rel L2 error | entries >10% off | flushed to zero |")
    a("|---|---|---|---|")
    a(f"| bf16 | {r['grad_relerr_bf16']:.2e} | **{r['grad_bad_bf16']}%** | "
      f"**{r['grad_zeroed_bf16']}%** |")
    a(f"| fp16 | {r['grad_relerr_fp16']:.2e} | {r['grad_bad_fp16']}% | "
      f"**{r['grad_zeroed_fp16']}%** |")
    a(f"| fp8 E4M3 | {r['grad_relerr_fp8']:.2e} | {r['grad_bad_fp8']}% | "
      f"**{r['grad_zeroed_fp8']}%** |")
    a(f"| fp8 E4M3, per-tensor scaled | {r['grad_relerr_fp8_scaled']:.2e} | "
      f"{r['grad_bad_fp8_scaled']}% | **{r['grad_zeroed_fp8_scaled']}%** |")
    a("")
    a(f"Read the last column, not the first. fp16 has the *lowest* L2 error of the three — "
      f"it has more mantissa bits than bf16 — and still sends "
      f"**{r['grad_zeroed_fp16']}% of the gradient entries to exactly zero**, because L2 "
      f"error is set by the largest entries while the small ones are where the rare-token "
      f"updates live. bf16 loses nothing: **{r['grad_zeroed_bf16']}% flushed**, "
      f"{r['grad_bad_bf16']}% of entries meaningfully wrong. And fp8 E4M3 destroys "
      f"{r['grad_zeroed_fp8']}% of the gradient unscaled, still "
      f"{r['grad_zeroed_fp8_scaled']}% with per-tensor scaling — the range simply is not "
      f"there, and making fp8 work for gradients needs finer-grained scaling than one "
      f"factor per tensor.")
    a("")
    a("![formats](results/formats.png)")
    a("")
    a("And the decision measured rather than argued — the same model, same seed, same "
      "batches, trained twice:")
    a("")
    a("| | loss, mean of last 20 steps | wall clock |")
    a("|---|---|---|")
    a(f"| fp32 (TF32 matmuls) | {r['prec_fp32_final']:.4f} | {r['prec_fp32_secs']}s |")
    a(f"| bf16 autocast | {r['prec_bf16_final']:.4f} | {r['prec_bf16_secs']}s "
      f"(**{r['prec_speedup']}×**) |")
    a("")
    a(f"**{r['prec_loss_delta']:+.4f} nats** for a {r['prec_speedup']}× speedup, on a GPU "
      f"whose bf16 tensor cores are {r['bf16_over_fp32']}× its fp32 pipeline. fp8 stays off "
      f"the table here for a second reason as well: this card is "
      f"`sm_{r['gpu_cc'].replace('.', '')}`, which has no fp8 tensor cores at all, so every "
      f"fp8 number above is a software cast — the numerical consequence without the speed "
      f"that would justify paying it.")
    a("")
    a("**And what stays in fp32, because the answer is not \"bf16 everywhere\":** the master "
      "weights and Adam's two moments. A weight update is a small number added to a large "
      "one, and bf16's 8 mantissa bits resolve about 0.4% relative — an update smaller than "
      "that vanishes entirely when the sum is rounded back. So bf16 activations and "
      f"matmuls, fp32 accumulation inside them, fp32 master weights, fp32 optimizer state. "
      f"That is exactly the {r['bytes_per_param']} bytes per parameter counted in §1, and it "
      f"is what the extra bytes are buying.")
    a("")
    a(f"*Produced by `truth_harness.ipynb` on {r.get('gpu') or 'CPU'} "
      f"({r.get('gpu_sms', '?')} SMs, {r.get('gpu_total_gib', '?')} GiB), torch "
      f"{r['torch_version']}, seed {r['seed']}, in {r['elapsed_s'] / 60:.1f} minutes. All "
      f"{r['n_results_keys']} recorded values are read out of "
      f"[`results/results.json`](results/results.json) by `tools/render_numbers.py`. GPU "
      f"float reductions are not bit-reproducible and this laptop's clocks move with "
      f"temperature, so a re-run moves the last decimals and every timing; the JSON, the "
      f"log and this table are always regenerated together.*")
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
