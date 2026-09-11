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


def mean(xs):
    return sum(xs) / len(xs)


def spread(xs):
    return max(xs) - min(xs)


def render(r):
    L = []
    a = L.append

    # ------------------------------------------------------------------ 1. Adam by hand
    a("## 1. Adam by hand, and the same five steps through PyTorch")
    a("")
    a(f"One weight at `w = {r['hand_w0']}`, five gradients "
      f"`{', '.join(str(g) for g in r['hand_grads'])}`, `lr = {r['hand_lr']:.0e}`, "
      f"`betas = ({r['hand_b1']}, {r['hand_b2']})`, `eps = {r['hand_eps']:.0e}`, all in "
      f"float64. The gradients are chosen to exercise the machinery rather than to look "
      f"tidy: a sign flip at step 3, and a {r['outlier_g_ratio']:.0f}× outlier at step 4.")
    a("")
    a("| t | g | m | v | m̂ | v̂ | step | w |")
    a("|---|---|---|---|---|---|---|---|")
    for row in r["hand_rows"]:
        a(f"| {int(row['t'])} | {row['g']:.2f} | {row['m']:.9f} | {row['v']:.10f} | "
          f"{row['mhat']:.9f} | {row['vhat']:.10f} | {row['step']:.10f} | "
          f"{row['w']:.12f} |")
    a("")
    a(f"`torch.optim.Adam`, fed the same five gradients by assignment, agrees on **`m`, `v` "
      f"and `w` at every one of the five steps to {r['hand_vs_torch_worst_digits']:.0f} "
      f"significant digits** — the largest disagreement on the weight is "
      f"`{r['hand_vs_torch_maxdiff_w']:.1e}`. Torch never materialises `m̂` or `v̂` (it folds "
      f"both corrections into the step, in a different algebraic order than the paper writes "
      f"them), so the check is made against the two tensors it does keep, `exp_avg` and "
      f"`exp_avg_sq`, plus the weight.")
    a("")
    a("**Three things the rows say that the formula does not.**")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| the first step is `lr`, whatever the gradient | `|step₁|/lr = "
      f"{r['first_step_over_lr']:.8f}` — at `t=1`, `m̂ = g` and `v̂ = g²`, so the step is "
      f"`lr·sign(g)` and the gradient's size cancels entirely |")
    a(f"| a {r['outlier_g_ratio']:.0f}× gradient does not buy a "
      f"{r['outlier_g_ratio']:.0f}× step | it buys **{r['outlier_step_ratio']:.2f}×**. "
      f"This is the whole reason Adam is used on language models |")
    a(f"| and the price of it | the spike stays in `v` for ~`1/(1-β₂)` = "
      f"{r['v_window']:,} steps, shrinking every step taken after it |")
    a("")
    a(f"**The scale invariance, tested rather than believed.** Multiply every gradient by "
      f"1000 and the trajectory is unchanged to "
      f"{r['scale_inv_digits_up']:.0f} digits (largest difference "
      f"`{r['scale_inv_drift_up']:.1e}`). Multiply by 1e-6 and it moves by "
      f"`{r['scale_inv_drift_dn']:.1e}` — only {r['scale_inv_digits_dn']:.0f} digits "
      f"survive. `eps` sits in the denominator next to `√v̂`, so when the gradients get small "
      f"enough that `√v̂` approaches `eps`, the scale stops cancelling and Adam degenerates "
      f"toward SGD. That is the failure mode `eps` is usually described as *preventing*, and "
      f"it is also the one it causes.")
    a("")
    a(f"**Adam + L2 is not AdamW, and five steps is enough to show it.** At λ = "
      f"{r['wd_lambda']}, same five gradients:")
    a("")
    a("| | weight after five steps | agrees with torch to |")
    a("|---|---|---|")
    a(f"| `Adam(weight_decay=λ)` — decay through the gradient | "
      f"`{r['wd_adam_l2_hand']:.16f}` | {r['wd_adam_l2_digits']:.0f} digits |")
    a(f"| `AdamW(weight_decay=λ)` — decoupled | `{r['wd_adamw_hand']:.16f}` | "
      f"{r['wd_adamw_digits']:.0f} digits |")
    a(f"| no decay at all | `{r['hand_w_final']:.16f}` | — |")
    a("")
    a(f"The two decay rules differ by `{r['wd_gap']:.2e}` after five steps: coupled L2 flows "
      f"through `m` and `v` and is divided by `√v̂` along with everything else, so a "
      f"parameter with large gradients gets decayed *less*, which is the opposite of what a "
      f"regulariser is for.")
    a("")
    a(f"And a detail that only turns up if you insist on all 16 digits: **torch shrinks `w` "
      f"*before* taking the Adam step**, while the AdamW paper's Algorithm 2 subtracts both "
      f"from the same `w`. Decaying last instead moves the answer by "
      f"`{r['wd_order_gap']:.2e}` — `O(η²λ)` per step, invisible at four decimal places, and "
      f"exactly how a reimplementation of AdamW ends up being subtly not AdamW.")
    a("")

    # ------------------------------------------------------------------ 0. the floor
    a("## The noise floor, measured before anything is compared")
    a("")
    a(f"Sections 2, 4 and 6 all end in \"is this difference real?\", so the answer is "
      f"measured once, up front. Two things that *should not* matter are varied on an "
      f"otherwise identical {r['noise_steps']}-step run:")
    a("")
    a("| what was varied | mean held-out loss | sd | spread |")
    a("|---|---|---|---|")
    a(f"| 5 data orders, same init | {mean(r['noise_data_losses']):.4f} | "
      f"{r['noise_data_sd']:.4f} | {r['noise_data_spread']:.4f} |")
    a(f"| 5 inits, same data order | {mean(r['noise_init_losses']):.4f} | "
      f"{r['noise_init_sd']:.4f} | {r['noise_init_spread']:.4f} |")
    a("")
    a(f"**Noise floor = {r['noise_floor']:.4f} nats.** Every verdict below is stated against "
      f"it, and a difference smaller than it is reported as no result rather than a small "
      f"one.")
    a("")
    a(f"The optimizer that sections 2 and 3 run on is written from scratch, because "
      f"`torch.optim` will not let you switch bias correction off. Fed a fixed sequence of 60 "
      f"gradients in float64 with no model in the loop, it agrees with `torch.optim.AdamW` "
      f"to `{r['openloop_maxdiff']:.1e}` — the same algorithm, to the limit of float64. Put "
      f"the same two optimizers *inside* a training loop in fp32 and after 40 steps the "
      f"weights are **`{r['handadamw_vs_torch_param_rel']:.1e}` apart**, while the same code "
      f"run twice is bit-identical (`{r['rerun_param_rel']:.1e}`). Two mathematically "
      f"identical optimizers, one ulp of difference in the order of two divisions, and 40 "
      f"steps of feedback to amplify it. That is what the noise floor is made of.")
    a("")

    # ------------------------------------------------------------------ 2. bias correction
    a("## 2. Bias correction off — the first twenty steps, both ways")
    a("")
    a("Putting both hat terms into the step cancels everything except one scalar:")
    a("")
    a("```")
    a("m̂ / √v̂  =  (m / √v) × r(t),     r(t) = √(1 - β₂ᵗ) / (1 - β₁ᵗ)")
    a("```")
    a("")
    a(f"So switching bias correction off multiplies **every parameter's step by `1/r(t)`**, "
      f"the same number for all {r['n_params_base']:,} of them. The first twenty:")
    a("")
    rows = r["bc_rows20"]
    a("| t | " + " | ".join(str(int(t)) for t, _ in rows[:10]) + " |")
    a("|---|" + "---|" * 10)
    a("| step is this much too large | "
      + " | ".join(f"{1 / v:.2f}×" for _, v in rows[:10]) + " |")
    a("")
    a("| t | " + " | ".join(str(int(t)) for t, _ in rows[10:]) + " |")
    a("|---|" + "---|" * 10)
    a("| step is this much too large | "
      + " | ".join(f"{1 / v:.2f}×" for _, v in rows[10:]) + " |")
    a("")
    a(f"**The usual description of bias correction is wrong, and the table says so.** It is "
      f"not that the *first* step would be huge. `1 - β₁ᵗ` recovers on a ~10-step timescale "
      f"and `1 - β₂ᵗ` on a ~1,000-step one, so early on the two corrections pull in opposite "
      f"directions and the damage **peaks at step {r['bc_worst_t']}**, where the uncorrected "
      f"step is **{r['bc_worst_factor']:.2f}× too large** — against "
      f"{1 / r['bc_r1']:.2f}× at step 1. Over the first twenty steps that compounds into "
      f"{r['bc_scalar_travel_ratio']:.2f}× the distance travelled on a constant gradient.")
    a("")
    a(f"On the real model, the scalar is confirmed by asking the corrected run, at every "
      f"step, what step it *would* have taken from the same `m` and `v` with the hats "
      f"removed — same weights, so nothing but the correction is being measured. That "
      f"counterfactual reproduces `1/r(t)` to **{r['bc_cf_err_pct']:.2f}%** at every one of "
      f"{r['bc_steps']:,} steps.")
    a("")
    a("![bias correction](results/bias_correction.png)")
    a("")
    a("### After how many steps does it stop mattering?")
    a("")
    a("Three defensible criteria, three different answers, and the question is not answerable "
      "without saying which one is meant.")
    a("")
    never = "**never**, in {:,} steps".format(r["bc_steps"])
    b_cell = ("step {:,}".format(r["bc_answer_B_10pct"])
              if r["bc_answer_B_10pct"] <= r["bc_steps"] else never)
    c_cell = ("step {:,}".format(r["bc_answer_C_step"])
              if r["bc_answer_C_step"] else never)
    a("| criterion | within 10% | within 1% |")
    a("|---|---|---|")
    a(f"| **(A) the step** — `r(t)` is a property of the betas alone | step "
      f"**{r['bc_answer_A_10pct']:,}** | step **{r['bc_answer_A_1pct']:,}** |")
    a(f"| **(B) the two runs' step sizes** | {b_cell} | — |")
    a(f"| **(C) the model** — held-out loss, against the {r['noise_floor']:.4f}-nat floor | "
      f"{c_cell} | — |")
    a("")
    a("Criterion (B) is the one that looks like the obvious measurement and is the wrong "
      "question: after a handful of steps the two arms are different models seeing the same "
      "data, so their step sizes are measuring divergence, not correction.")
    a("")
    a("Criterion (C) is the one that matters:")
    a("")
    a("| held-out loss | corrected | uncorrected | gap |")
    a("|---|---|---|---|")
    for s, on, off in zip(r["bc_eval_steps"], r["bc_eval_on"], r["bc_eval_off"]):
        if s in (25, 100, 400, r["bc_steps"]):
            a(f"| step {s:,} | {on:.4f} | {off:.4f} | **{off - on:+.4f}** |")
    a("")
    a(f"The gap **peaks at {r['bc_peak_gap']:+.4f} nats at step {r['bc_peak_gap_step']}** "
      f"({abs(r['bc_peak_gap']) / r['noise_floor']:.0f}× the noise floor) and is still "
      f"{r['bc_final_gap']:+.4f} ({abs(r['bc_final_gap']) / r['noise_floor']:.0f}×) at step "
      f"{r['bc_steps']:,}. In parameter space the two models end "
      f"`‖w_off − w_on‖/‖w_on‖ = {r['bc_param_distance']:.2f}` apart, with the uncorrected "
      f"run's weights **{r['bc_weight_norm_ratio']:.2f}× larger** — the first few dozen steps "
      f"were up to {r['bc_worst_factor']:.1f}× too big and nothing gives that back.")
    a("")
    a(f"**So the answer is: the step stops differing by more than 10% after ~"
      f"{r['bc_answer_A_10pct']:,} steps and by more than 1% after ~{r['bc_answer_A_1pct']:,}"
      f" — but the model never recovers.** Bias correction is not a transient. It is a "
      f"permanent difference introduced during a transient.")
    a("")
    a("### Unless you have warmup, which does most of the same job")
    a("")
    a(f"Bias correction shrinks the first few dozen steps; so does warmup. If warmup were "
      f"enough, nobody would need the hat terms — twenty seconds of GPU time settles it.")
    a("")
    a("| | corrected | uncorrected | gap |")
    a("|---|---|---|---|")
    a(f"| no warmup, at step {r['bc_steps']:,} | {r['bc_nowu_final_on']:.4f} | "
      f"{r['bc_nowu_final_off']:.4f} | **{r['bc_nowu_final_off'] - r['bc_nowu_final_on']:+.4f}** |")
    a(f"| {r['bc_warmup_len']}-step warmup | {r['bc_wu_final_on']:.4f} | "
      f"{r['bc_wu_final_off']:.4f} | **{r['bc_wu_final_off'] - r['bc_wu_final_on']:+.4f}** |")
    a("")
    a(f"Warmup cuts the largest step the uncorrected run ever takes by "
      f"**{r['bc_wu_suppression']:.1f}×**, and the final gap shrinks "
      f"**{r['bc_wu_gap_shrink']:.1f}×**, from {r['bc_nowu_gap_final']:+.4f} to "
      f"{r['bc_wu_gap_final']:+.4f}. Over the second half of the run the warmed-up gap sits "
      f"inside the noise floor {r['bc_wu_inside_floor_frac'] * 100:.0f}% of the time, against "
      f"0% without warmup.")
    a("")
    a(f"Stated carefully, because it only just clears: with warmup the penalty for dropping "
      f"bias correction falls to **{abs(r['bc_wu_gap_final']) / r['noise_floor']:.1f}× the "
      f"noise floor** — the edge of what this experiment can resolve — while without warmup "
      f"it is {abs(r['bc_nowu_gap_final']) / r['noise_floor']:.0f}× and never in doubt. So "
      f"**warmup largely substitutes for bias correction**: not by making the early steps "
      f"correct, but by making them small enough that being {r['bc_worst_factor']:.1f}× too "
      f"large costs little. That is why a modern recipe gets away with either, and why the "
      f"two are so often confused — different mechanisms, aimed at the same hundred steps.")
    a("")

    # ------------------------------------------------------------------ 3. ratios
    a("## 3. The update-to-weight ratio, every layer, every step")
    a("")
    a(f"`RMS(Δw)/RMS(w)`, logged for all {len(r['ratio_layers'])} parameter tensors at every "
      f"one of {r['ratio_steps']} steps, with a {r['ratio_warmup']}-step warmup and then a "
      f"constant learning rate, so that warmup is the only schedule feature in play.")
    a("")
    a("| layer | step 1 | step 10 | step 50 | step 100 | step 300 | step 600 |")
    a("|---|---|---|---|---|---|---|")
    for name in r["ratio_layers"]:
        a(f"| `{name}` | " + " | ".join(f"{v:.2e}" for v in r["ratio_table"][name]) + " |")
    a("")
    a(f"**Read the `1.00e+00` entries at step 1 before anything else.** "
      f"{len(r['ratio_zero_init'])} tensors report a ratio of exactly 1 on the first step: "
      f"the LayerNorm shifts, which are initialised to zero, so at step 1 the weight *is* the "
      f"update. That is an artefact of the initialisation rather than a layer moving fast, "
      f"and it is why the spread below is quoted over the matrices and embeddings — the "
      f"tensors where `RMS(w)` means something at step 1.")
    a("")
    a(f"At the end of warmup those span **{r['ratio_spread']:.0f}×** — slowest "
      f"`{r['ratio_slowest']}` at {r['ratio_peak'][r['ratio_slowest']]:.2e}, fastest "
      f"`{r['ratio_fastest']}` at {r['ratio_peak'][r['ratio_fastest']]:.2e}. All of them sit "
      f"roughly ten times above the `1e-3` rule of thumb, which is a fact about this "
      f"learning rate rather than about the model: `{r['ratio_lr']:.0e}` held constant with "
      f"no decay is close to the *peak* of a tuned schedule (§5 puts the tuned optimum at "
      f"this width near {r['sweep_minima']['sp_' + str(r['base_width'])]:.1e}), and the "
      f"folklore figure describes a run that has already cooled down.")
    a("")
    a(f"And the thing the word *plateau* would have hidden: **with the learning rate held "
      f"constant after step {r['ratio_warmup']}, the ratio does not stay put.** It peaks at "
      f"the end of warmup and then decays — a median of "
      f"**{r['ratio_decay']:.2f}×** by step {r['ratio_steps']} — because Adam pins `RMS(Δw)` "
      f"near `η` while `RMS(w)` keeps growing. A constant learning rate is not a constant "
      f"update-to-weight ratio, and the ratio is the one of the two that describes what the "
      f"model is doing.")
    a("")
    a("![update ratio](results/update_ratio.png)")
    a("")
    a("### The step at which warmup stops changing it")
    a("")
    a("The ratio factorises, and that makes the question exact rather than a matter of "
      "eyeballing a curve:")
    a("")
    a("```")
    a("ρ(t) = η(t) × ρ̃(t),     ρ̃(t) = RMS(m̂/(√v̂+ε)) / RMS(w)")
    a("```")
    a("")
    a(f"Warmup enters **only** through `η(t)`. So it stops changing the ratio at exactly the "
      f"last warmup step, step {r['ratio_warmup']} — after which `Δ log η = 0` and warmup's "
      f"contribution is identically zero. That is the literal answer, and it is true by "
      f"construction rather than by measurement.")
    a("")
    a(f"The measurable question is the other half: *during* warmup, is warmup even what is "
      f"moving the ratio? Differencing the logs splits each step's movement into warmup's "
      f"share and everything else's, in the same units — and the answer is the opposite of "
      f"what reading the schedule suggests. Warmup's per-step contribution is "
      f"`log((t+1)/t)`, which is 0.69 at step 2 and falls like `1/t`; Adam's is roughly flat.")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| warmup is the larger term | steps 2 to **{r['ratio_warmup_dominant_until']}** |")
    a(f"| Adam's own transient is the larger term | step "
      f"{r['ratio_warmup_dominant_until'] + 1} to the end of warmup at {r['ratio_warmup']} |")
    a(f"| warmup's share of the total movement across warmup | "
      f"**{r['ratio_warmup_share'] * 100:.0f}%**, almost all of it in those first "
      f"{r['ratio_warmup_dominant_until']} steps |")
    a("")
    a(f"**So the answer has two halves and they are different numbers.** Warmup stops "
      f"*changing* the ratio at step {r['ratio_warmup']}, by construction, since `η` stops "
      f"changing. Warmup stops being what *drives* the ratio at step "
      f"**{r['ratio_warmup_dominant_until']}** — measured, and "
      f"{r['ratio_warmup'] // max(r['ratio_warmup_dominant_until'], 1)}× earlier than the "
      f"schedule would lead you to believe. For most of a 100-step warmup, the thing moving "
      f"the update-to-weight ratio is Adam's own bias transient: section 2's `r(t)`, arriving "
      f"from a different direction.")
    a("")
    a("### Does the ratio catch a wrong parameterisation? No — and that is worth knowing")
    a("")
    a(f"The ratio is routinely recommended as *the* diagnostic for whether a model is "
      f"parameterised correctly at a new width, which is section 5's subject. Four short runs "
      f"say it is not one, and the reason is arithmetic: μP shrinks the update **and** the "
      f"initialisation, and the ratio only sees their quotient. `RMS(Δw) ∝ η/m` over "
      f"`RMS(w) ∝ 0.02/√m` gives `∝ 1/√m`, so μP's ratio is *supposed* to drift.")
    a("")
    a("| hidden matrices, median ratio after warmup | width 256 | width 1024 | measured | theory |")
    a("|---|---|---|---|---|")
    a(f"| standard parameterisation | {r['ratio_cmp']['sp_256']:.2e} | "
      f"{r['ratio_cmp']['sp_1024']:.2e} | **{r['ratio_drift_sp']:.2f}×** | "
      f"{r['ratio_drift_sp_pred']:.2f}× |")
    a(f"| μP | {r['ratio_cmp']['mup_256']:.2e} | {r['ratio_cmp']['mup_1024']:.2e} | "
      f"**{r['ratio_drift_mup']:.2f}×** | {r['ratio_drift_mup_pred']:.2f}× |")
    a("")
    a(f"{r['ratio_drift_sp']:.2f}× against {r['ratio_drift_mup']:.2f}× where the theory asks "
      f"for a factor of two between them. μP lands on its prediction; **SP misses its own "
      f"badly**, and the reason is worth more than the check: at this learning rate, width "
      f"1024 is far past SP's optimum (§5 puts it near 5e-04 there), so that arm is training "
      f"badly and its weights are growing for reasons that have nothing to do with "
      f"parameterisation. The diagnostic is confounded by exactly the condition it is "
      f"supposed to detect.")
    a("")
    a(f"**Conclusion, against the folklore:** the update-to-weight ratio is a good instrument "
      f"for *is this layer moving at a sane speed* and a bad one for *is this model "
      f"parameterised correctly at a new width*. The quantity that answers the second "
      f"question is the scale of the activations, and §5 measures it.")
    a("")

    # ------------------------------------------------------------------ 4. schedules
    a("## 4. Cosine against WSD, both tuned, both stopped at 200")
    a("")
    a(f"Four arms, each with its own peak learning rate chosen from the **same "
      f"{len(r['sched_lr_grid'])}-point grid** — a schedule comparison at a single shared "
      f"learning rate is a comparison of learning rates — and each run three times with "
      f"different data orders.")
    a("")
    a("| arm | tuned peak lr | loss at step 200 | loss at its own end |")
    a("|---|---|---|---|")
    for name in ("cosine, budget 300", "WSD, budget 300", "cosine, budget 200",
                 "WSD cooled into 200"):
        at2, ate = r["sched_arm_at_200"][name], r["sched_arm_at_end"][name]
        key = {"cosine, budget 300": "cos300", "WSD, budget 300": "wsd300",
               "cosine, budget 200": "cos200", "WSD cooled into 200": "wsd200"}[name]
        a(f"| {name} | {r['sched_best_lrs'][key]:.0e} | **{mean(at2):.4f}** "
          f"[±{spread(at2) / 2:.3f}] | {mean(ate):.4f} [±{spread(ate) / 2:.3f}] |")
    a("")
    c3 = mean(r["sched_arm_at_200"]["cosine, budget 300"])
    w3 = mean(r["sched_arm_at_200"]["WSD, budget 300"])
    w2 = mean(r["sched_arm_at_200"]["WSD cooled into 200"])
    best = r["sched_verdict_best_at_200"]
    lit = r["sched_literal_gap"]
    if r["sched_literal_resolved"]:
        a(f"**The literal answer.** Both budgeted for 300, both stopped at "
          f"{r['sched_stop_at']}: cosine **{c3:.4f}**, WSD **{w3:.4f}**, a gap of "
          f"**{lit:+.4f} nats** — {abs(lit) / r['noise_floor']:.1f}× the noise floor, so "
          f"{'cosine' if c3 < w3 else 'WSD'} wins by a resolvable margin.")
    else:
        a(f"**The literal answer is that there isn't one.** Both budgeted for 300, both "
          f"stopped at {r['sched_stop_at']}: cosine **{c3:.4f}**, WSD **{w3:.4f}** — a gap "
          f"of **{lit:+.4f} nats** against a noise floor of {r['noise_floor']:.4f}. That is "
          f"**{abs(lit) / r['noise_floor']:.1f}× the floor**: the two arms are separated by "
          f"less than the spread of runs that differ only in their data order. Run carefully, "
          f"with three seeds a side and both peak learning rates tuned, the comparison the "
          f"assignment asks for **does not have an answer at this scale**. Reporting "
          f"\"{'WSD' if w3 < c3 else 'cosine'} wins by {abs(lit):.4f}\" would be reporting a "
          f"coin flip — which is §6's lesson arriving one section early.")
    a("")
    a(f"**And why stopping both at the same step was never going to settle it.** At step "
      f"{r['sched_stop_at']} the cosine arm has already decayed to "
      f"{r['sched_lr_frac_at_200']['cosine, budget 300'] * 100:.0f}% of its peak while the "
      f"WSD arm is still at "
      f"{r['sched_lr_frac_at_200']['WSD, budget 300'] * 100:.0f}%. A model sitting at a high "
      f"learning rate is mid-exploration and its loss is *supposed* to look worse; a model "
      f"that has decayed has consolidated and stopped exploring. Cosine's formula contains "
      f"`T`, so declaring 300 and stopping at 200 catches it two-thirds of the way down a "
      f"ramp aimed somewhere else. Stopping both at the same step compares two different "
      f"stages of two different plans.")
    a("")
    a("![schedules](results/schedules.png)")
    a("")
    a("### Which model I would keep")
    a("")
    c200 = mean(r["sched_arm_at_200"]["cosine, budget 200"])
    a(f"1. **Of the two arms the assignment names: neither** — and not because I dislike the "
      f"question. {min(c3, w3):.4f} against {max(c3, w3):.4f} is "
      f"{abs(lit) / r['noise_floor']:.1f}× the noise floor. Forced to pick, I would take the "
      f"{'cosine' if c3 < w3 else 'WSD'} checkpoint and I would not defend the choice.")
    a(f"2. **What I would actually keep is *{best}*** at "
      f"{mean(r['sched_arm_at_200'][best]):.4f} — "
      f"{c3 - mean(r['sched_arm_at_200'][best]):+.4f} nats better than the stopped cosine "
      f"arm, {abs(r['sched_wsd_cooled_vs_cosine']) / r['noise_floor']:.1f}× the noise floor, "
      f"and available for the same {r['sched_stop_at']} steps of compute. WSD's cooldown is "
      f"aimed at the step you intend to stop on, and aiming it correctly is worth more here "
      f"than the choice of schedule family.")
    a(f"3. **And the arm that did not work, because it is the one I expected to win.** "
      f"Cosine re-planned for a {r['sched_stop_at']}-step budget is the *worst* of the four "
      f"at {c200:.4f}, despite being tuned over the same grid and picking a higher peak "
      f"({r['sched_best_lrs']['cos200']:.0e}) to compensate. A full cosine decay inside "
      f"{r['sched_stop_at']} steps spends too much of a short run at a low learning rate; the "
      f"arm that was \"interrupted\" had simply taken bigger steps for longer. At this "
      f"horizon, being caught mid-decay beats having decayed — which is not what the tidy "
      f"version of this section would have said.")
    a("")
    end_w = mean(r["sched_arm_at_end"]["WSD, budget 300"])
    end_c = mean(r["sched_arm_at_end"]["cosine, budget 300"])
    resolved = abs(r["sched_end_gap"]) > r["noise_floor"]
    a(f"**And does WSD beat cosine?** At its own 300-step horizon WSD reaches {end_w:.4f} "
      f"against cosine's {end_c:.4f} — {r['sched_end_gap']:+.4f} nats, "
      f"{abs(r['sched_end_gap']) / r['noise_floor']:.1f}× the noise floor. "
      + (f"On this run **{'WSD' if r['sched_end_gap'] < 0 else 'cosine'} wins** and the "
         f"margin is resolvable."
         if resolved else
         "That is inside the noise floor, so this run does not answer it."))
    a("")
    a(f"**What that does not license is the general claim.** This is "
      f"{r['n_params_base'] / 1e6:.1f}M parameters, {r['sched_steps']} steps, three seeds, "
      f"one depth, one batch size, and a stable phase about "
      f"{r['sched_steps'] - int(0.2 * r['sched_steps']) - r['sched_warmup']} steps long. The "
      f"published WSD results are about runs where that phase is thousands of times longer, "
      f"and the mechanism usually offered for why it works — that a long high-`η` phase finds "
      f"wider basins — cannot be tested at this scale at all. A result that agrees with the "
      f"literature for reasons the experiment cannot check is still one data point.")
    a("")
    a(f"What this run *does* establish is narrower and more useful: WSD's cooldown can be "
      f"aimed at a stopping point chosen after the run started, and doing so is worth "
      f"{r['sched_wsd_cooled_vs_cosine']:+.4f} nats against a cosine schedule that had to "
      f"commit to its horizon in advance.")
    a("")

    # ------------------------------------------------------------------ 5. width sweep
    a("## 5. Learning rate against width, at 256, 512 and 1024")
    a("")
    a(f"{len(r['sweep_lrs'])} learning rates × {len(r['sweep_widths'])} widths × 2 "
      f"parameterisations, {r['sweep_steps']} steps each. Run twice because the whole point is "
      f"the extrapolation, and under standard parameterisation the extrapolation is a guess: "
      f"SP gives every tensor the same `η`, so as the model widens that same `η` is driving a "
      f"different model. μP initialises hidden and readout matrices `∝ 1/√fan_in` and gives "
      f"them `η/m`, leaving embeddings and LayerNorm gains at `η`, and claims the optimum then "
      f"stops moving. That is falsifiable, so this section tries to falsify it.")
    a("")
    same = ("agree to the last bit: every one of the losses is identical"
            if r["sweep_base_identical"] == 0
            else f"agree to `{r['sweep_base_identical']:.0e}`")
    a(f"At the base width `m = 1`, so the two parameterisations are *the same model with the "
      f"same learning rates* — the width-{r['base_width']} rows are a control, and they "
      f"{same}. If they had not, the μP implementation would be wrong and everything below "
      f"it meaningless.")
    a("")
    a("| | " + " | ".join(f"{lr:.1e}" for lr in r["sweep_lrs"]) + " | optimum |")
    a("|---|" + "---|" * (len(r["sweep_lrs"]) + 1))
    for pz in ("sp", "mup"):
        for w in r["sweep_widths"]:
            cur = r["sweep_curves"][f"{pz}_{w}"]
            lo = min(cur)
            cells = " | ".join(f"**{v:.3f}**" if v == lo else f"{v:.3f}" for v in cur)
            a(f"| {'SP' if pz == 'sp' else 'μP'}, width {w} | {cells} | "
              f"{r['sweep_minima'][f'{pz}_{w}']:.2e} |")
    a("")
    a(f"The optima are refined off the factor-of-two grid by fitting a parabola through the "
      f"best point and its two neighbours. **Across a 4× change in width the optimum moves "
      f"{r['sweep_sp_shift']:.2f}× under SP and {r['sweep_mup_shift']:.2f}× under μP.**")
    a("")
    a("![lr sweep](results/lr_sweep.png)")
    a("")
    a("### The mechanism, checked rather than cited")
    a("")
    a(f"μP transfers because it is built so the scale of what flows through the network does "
      f"not change with width. If the implementation is right, coordinates stay put as the "
      f"model widens — at init *and* after a few steps, which is the part that is easy to get "
      f"wrong:")
    a("")
    a("| RMS, width 1024 ÷ width 256 | SP | μP |")
    a("|---|---|---|")
    a(f"| residual stream, at init | {r['coord_growth']['sp_resid_0']:.2f}× | "
      f"{r['coord_growth']['mup_resid_0']:.2f}× |")
    a(f"| logits, at init | {r['coord_growth']['sp_logits_0']:.2f}× | "
      f"{r['coord_growth']['mup_logits_0']:.2f}× |")
    a(f"| residual stream, after 10 steps | {r['coord_growth']['sp_resid_10']:.2f}× | "
      f"{r['coord_growth']['mup_resid_10']:.2f}× |")
    a(f"| logits, after 10 steps | **{r['coord_growth']['sp_logits_10']:.2f}×** | "
      f"**{r['coord_growth']['mup_logits_10']:.2f}×** |")
    a("")
    a("Under SP a wider model puts a larger number into the loss, so the same learning rate "
      "is effectively a larger one, so the optimum has to move. That is the whole mechanism.")
    a("")
    a("### The value I would use at width 4096, and how much I believe it")
    a("")
    a(f"**{r['final_lr_4096']:.1e} under μP**, with the width multiple `m = 4096/256 = 16` "
      f"applied inside the optimizer — so the hidden and readout matrices actually see "
      f"`{r['final_lr_4096_matrices']:.2e}` and the embeddings and LayerNorm gains see "
      f"`{r['final_lr_4096']:.1e}`.")
    a("")
    a(f"The whole section is two numbers — the exponents of `lr*` against width, fitted over "
      f"the three widths. Zero means the tuning transfers; anything else means it has to be "
      f"redone at every width. μP's theory for Adam predicts **−1 for SP** and **0 for μP**:")
    a("")
    a("| | fitted exponent | predicted | at width 4096 |")
    a("|---|---|---|---|")
    a(f"| SP | **{r['sweep_sp_exponent']:+.2f}** | −1 | {r['sweep_sp_pred_4096']:.1e} |")
    a(f"| μP | **{r['sweep_mup_exponent']:+.2f}** | 0 | {r['final_lr_4096']:.1e} |")
    a("")
    a(f"Under SP the answer is a power law fitted to three grid-quantised optima and extended "
      f"two doublings past the last one measured. I would not use it — and the distance "
      f"between {r['sweep_sp_exponent']:+.2f} and the predicted −1 is itself a measure of how "
      f"much such an extrapolation is worth.")
    a("")
    a(f"**The prediction was then tested at a width not used to make it.** Width "
      f"{r['held_width']:,} was not in the sweep; each parameterisation's predicted optimum "
      f"was run there against its two factor-of-two neighbours:")
    a("")
    a("| | ÷2 | prediction | ×2 | prediction was best of three |")
    a("|---|---|---|---|---|")
    a(f"| SP, predicted {r['held_sp_grid'][1]:.2e} | {r['held_sp_loss'][0]:.4f} | "
      f"**{r['held_sp_loss'][1]:.4f}** | {r['held_sp_loss'][2]:.4f} | "
      f"{'yes' if r['held_sp_hit'] else 'no'} |")
    a(f"| μP, predicted {r['held_mup_grid'][1]:.2e} | {r['held_mup_loss'][0]:.4f} | "
      f"**{r['held_mup_loss'][1]:.4f}** | {r['held_mup_loss'][2]:.4f} | "
      f"{'yes' if r['held_mup_hit'] else 'no'} |")
    a("")
    a(f"Trusting the μP transfer at the held-out width cost **{r['held_mup_cost']:+.4f} "
      f"nats** against the best of the three; trusting the SP fit cost "
      f"**{r['held_sp_cost']:+.4f}**.")
    a("")
    a("**Confidence, in the order that matters.**")
    a("")
    a(f"* *What was tested.* The μP optimum moved {r['sweep_mup_shift']:.2f}× across a 4× "
      f"width range, and the prediction survived at a width held out from the fit. 4096 is "
      f"one further doubling, 16× the base width.")
    a(f"* *What was not.* Depth is fixed at {r['n_layer']} and only width moves — μP's "
      f"guarantees are about width, and depth transfer is a separate and weaker claim. The "
      f"horizon is {r['sweep_steps']} steps and the batch is {r['B']}×{r['T']} tokens; the "
      f"optimal learning rate depends on both and neither was swept. This is the best "
      f"learning rate *for this recipe*, not a constant of the architecture.")
    a(f"* *The width that was not run.* At `n_layer={r['n_layer']}`, width 4096 is 402M "
      f"parameters — 6.4 GiB of weights, gradients and Adam's two moments before a single "
      f"activation, against {r.get('gpu_total_gib', 4)} GiB of card. The last doubling is an "
      f"extrapolation in every case, and that is stated rather than worked around.")
    a("")
    if r["held_mup_confident"]:
        a(f"So: **high confidence that {r['final_lr_4096']:.1e} is within a factor of 2 of "
          f"the optimum at width 4096 for this recipe** — a factor of 2 being the resolution "
          f"the sweep itself has. The transfer was given a chance to fail at a width it had "
          f"never seen, and did not.")
    else:
        a(f"So: **moderate confidence, and lower than I expected to be writing.** The "
          f"transfer was given a chance to fail at width {r['held_width']:,} and it did — the "
          f"predicted learning rate was not the best of the three tried, costing "
          f"{r['held_mup_cost']:+.4f} nats. I would still start from "
          f"{r['final_lr_4096']:.1e} at 4096, because it is a far better starting point than "
          f"the SP fit, but I would not commit a long run to it without checking.")
    a("")
    a(f"Either way: no confidence in a third significant figure, and no claim at all about a "
      f"different depth, batch size or token budget. The practical form of that is that I "
      f"would run one three-point confirmation sweep at the target width before committing a "
      f"long run — exactly what the table above is — and that is affordable precisely because "
      f"μP turned it into three runs instead of {len(r['sweep_lrs'])}.")
    a("")

    # ------------------------------------------------------------------ 6. tuning
    a("## 6. Tune both sides")
    a("")
    a("> *Almost every optimizer claim that failed to replicate was a well tuned method "
      "measured against a badly tuned one.*")
    a("")
    a(f"The protocol, fixed in advance and applied identically to every arm: the same number "
      f"of trials ({len(r['cmp_adamw_grid'])}), grids spaced by a factor of two and "
      f"positioned so the winner is **interior** (an optimum at the edge of a grid means the "
      f"arm was not tuned, and the code checks — "
      f"{'both optima came out interior' if r['cmp_both_interior'] else 'AN OPTIMUM LANDED ON AN EDGE, so that arm is not fairly tuned'}"
      f"), the same budget and schedule, and a verdict only when the gap clears the noise "
      f"floor.")
    a("")
    a("### AdamW against SGD with momentum")
    a("")
    a("| the comparison | AdamW | SGD+m | gap |")
    a("|---|---|---|---|")
    a(f"| SGD run at AdamW's learning rate ({r['cmp_adamw_lr']:.0e}) | "
      f"{r['cmp_adamw_best']:.4f} | {r['cmp_sgd_at_adamw_lr']:.4f} | "
      f"**{r['cmp_sgd_at_adamw_lr'] - r['cmp_adamw_best']:+.4f}** |")
    a(f"| AdamW run at SGD's learning rate ({r['cmp_sgd_lr']:.3g}) | "
      f"{r['cmp_adamw_at_sgd_lr']:.4f} | {r['cmp_sgd_best']:.4f} | "
      f"**{r['cmp_sgd_best'] - r['cmp_adamw_at_sgd_lr']:+.4f}** |")
    a(f"| both tuned, {len(r['cmp_adamw_grid'])} trials each | {r['cmp_adamw_best']:.4f} | "
      f"{r['cmp_sgd_best']:.4f} | **{r['cmp_sgd_best'] - r['cmp_adamw_best']:+.4f}** |")
    a("")
    a(f"The careless comparison overstates AdamW's advantage by "
      f"**{r['cmp_overstatement']:.1f}×**. Run the other way round it *reverses* the result: "
      f"at SGD's learning rate, AdamW "
      f"{'loses to' if r['cmp_adamw_at_sgd_lr'] > r['cmp_sgd_best'] else 'still beats'} SGD. "
      f"Either direction is available to anyone who tunes only one side, which is why the "
      f"direction chosen tends to be the flattering one. The real gap, "
      f"{r['cmp_sgd_best'] - r['cmp_adamw_best']:+.4f} nats, is "
      f"{abs(r['cmp_sgd_best'] - r['cmp_adamw_best']) / r['noise_floor']:.1f}× the noise "
      f"floor: AdamW is genuinely better here, by much less than the careless comparison "
      f"says, and it took 14 runs rather than 2 to find out.")
    a("")
    a("### How to discover an optimizer that does not exist")
    a("")
    a(f"The comparison above is the obvious failure — SGD at Adam's learning rate is visibly "
      f"broken and somebody would catch it. The dangerous version is the one where the badly "
      f"tuned baseline looks *fine*. So: a challenger, `AdamW` with `β₂ = 0.95` instead of "
      f"`0.999`, a real change that real papers make. Tune the challenger over "
      f"{len(r['cmp_b95_grid'])} learning rates. Run the baseline at "
      f"`{r['cmp_default_lr']:.0e}` — not a strawman, but the number the lecture says "
      f"everyone starts from and nanoGPT's own default.")
    a("")
    a("| | held-out loss | at lr |")
    a("|---|---|---|")
    a(f"| challenger, `β₂=0.95`, tuned | {r['cmp_b95_best']:.4f} | {r['cmp_b95_lr']:.0e} |")
    a(f"| baseline, `β₂=0.999`, at the default | {r['cmp_baseline_default']:.4f} | "
      f"{r['cmp_default_lr']:.0e} |")
    a(f"| baseline, `β₂=0.999`, tuned on the same grid | **{r['cmp_adamw_best']:.4f}** | "
      f"{r['cmp_adamw_lr']:.0e} |")
    a("")
    a(f"**The paper I could have written:** *\"shortening the second-moment window improves "
      f"held-out loss by {r['cmp_fake_gap']:.4f} nats\"* — "
      f"{abs(r['cmp_fake_gap']) / r['noise_floor']:.0f}× the noise floor, reproducible on "
      f"demand, and false.")
    a("")
    real = r["cmp_real_gap"]
    if r["cmp_real_inside_floor"]:
        verdict = ("There is no effect here that this experiment can resolve — the entire "
                   "claimed improvement was the baseline's untuned learning rate.")
    elif real < 0:
        verdict = (f"**The claimed improvement does not merely vanish — it reverses.** With "
                   f"both sides tuned, `β₂=0.95` is {abs(real):.4f} nats *worse* than the "
                   f"baseline it was supposed to beat. The untuned comparison got the sign "
                   f"wrong, not just the size.")
    else:
        verdict = (f"There is a real effect in the claimed direction, and it is "
                   f"{r['cmp_fake_gap'] / real:.0f}× smaller than the untuned comparison "
                   f"says.")
    a(f"**The measurement:** {real:+.4f} nats, "
      f"{'inside' if r['cmp_real_inside_floor'] else 'outside'} the "
      f"{r['noise_floor']:.4f}-nat noise floor. " + verdict)
    a("")
    a(f"Nothing about the first claim requires dishonesty. The baseline ran, it converged, "
      f"its loss curve looks healthy, and `{r['cmp_default_lr']:.0e}` is what the field's own "
      f"folklore recommends. The only thing wrong with it is that a learning rate "
      f"{r['cmp_adamw_lr'] / r['cmp_default_lr']:.0f}× higher was worth "
      f"{r['cmp_baseline_default'] - r['cmp_adamw_best']:.4f} nats to that same baseline, and "
      f"nobody looked.")
    a("")
    a("![tuning](results/tuning.png)")
    a("")
    a(f"*Produced by `optimizer_harness.ipynb` on {r.get('gpu') or 'CPU'} "
      f"({r.get('gpu_sms', '?')} SMs, {r.get('gpu_total_gib', '?')} GiB), torch "
      f"{r['torch_version']}, seed {r['seed']}, in {r['elapsed_s'] / 60:.1f} minutes. All "
      f"{r['n_results_keys']} recorded values are read out of "
      f"[`results/results.json`](results/results.json) by `tools/render_numbers.py`, and the "
      f"log, the figures, the JSON and this table are always regenerated together from one "
      f"execution. Re-executing the whole notebook on this machine reproduced all "
      f"{r['n_results_keys']} values **exactly**, which is the strongest claim the hardware "
      f"allows: different hardware or a different torch build selects different kernels, and "
      f"§0 measures what that is worth — one ulp of difference becomes "
      f"{r['handadamw_vs_torch_param_rel']:.0e} of the weights in 40 steps, so on another "
      f"machine expect the noise floor, not the last decimal.*")
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
