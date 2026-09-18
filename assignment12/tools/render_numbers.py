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

MIB = 2 ** 20
GIB = 2 ** 30
NAMES = {0: "DP (ZeRO-0)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
STAGES = (0, 1, 2, 3)


def k(d, stage):
    """results.json round-trips integer dict keys as strings."""
    return d[str(stage)] if str(stage) in d else d[stage]


def mib(b):
    return b / MIB


def render(r):
    L = []
    a = L.append
    psi = r["psi"]
    world = r["world"]

    # ------------------------------------------------------- 1. the machine
    a("## 1. The 32 virtual GPUs, and a fabric that had to prove itself first")
    a("")
    a(f"`{world}` GPUs as `{r['n_nodes']}` nodes of `{r['gpus_per_node']}`, because that is "
      f"the shape of the hardware being modelled: NVLink at **{r['bw_intra'] / 1e9:.0f} "
      f"GB/s** inside a box, InfiniBand at **{r['bw_inter'] / 1e9:.0f} GB/s** between them — "
      f"**{r['bw_ratio']:.0f}× slower**, which is the fact the whole second half of this "
      f"turns on.")
    a("")
    a(f"Each GPU is a rank, a node, and a **memory ledger with a hard ceiling of "
      f"{r['gpu_capacity_mib']:.0f} MiB**. Every tensor a rank owns is created through "
      f"`mem.alloc(...)`, which adds the tensor's true byte count — asked of the tensor, not "
      f"assumed — to a running total, updates a high-water mark, and raises `DeviceOOM` when "
      f"the rank has run out of room. That ceiling is the most important design decision "
      f"here: it is what turns *\"ZeRO-3 fits a bigger model\"* from a claim into an "
      f"experiment, and §7 grows the model until each stage hits it.")
    a("")
    a(f"The ceiling is {r['gpu_capacity_mib']:.0f} MiB rather than 80 GiB because 32 replicas "
      f"of an 80 GiB-class model is several terabytes and this laptop has 16 GB. Every card "
      f"is scaled down by one factor of **{r['scale_to_80gib']:,}** and nothing else is "
      f"touched — the bandwidths are the real ones, and every quantity ZeRO is about is a "
      f"ratio, so it survives the scaling exactly. §10 pushes a measurement back up and "
      f"checks it against a number that was not produced here.")
    a("")
    a("### Why the collectives are implemented and not costed")
    a("")
    a("Three collectives carry all of data parallelism:")
    a("")
    a("| | what it does | who ends up with what |")
    a("|---|---|---|")
    a("| **all-reduce** | combines a value from every GPU | *every* GPU gets the whole "
      "combined result |")
    a(f"| **reduce-scatter** | combines it the same way | every GPU gets **one slice** — "
      f"1/{world} each |")
    a("| **all-gather** | the reverse | every GPU contributes its slice and leaves with the "
      "whole thing |")
    a("")
    a("and the identity everything else rests on:")
    a("")
    a("> **all-reduce = reduce-scatter + all-gather.**")
    a("")
    a("That is not a coincidence, it is how a ring all-reduce is *built*. Which means plain "
      "data parallelism was always paying for both halves — and ZeRO-1 and ZeRO-2 get their "
      "memory back **for free** by noticing that there is a moment between the two halves "
      "when every rank is holding exactly its own slice of the reduced gradient and nothing "
      "else. Everything those two stages do happens in that gap. This is the single most "
      "useful idea in the session, and it is why §5 comes out the way it does.")
    a("")
    a(f"So the ring is written out: `2(N−1)` steps of `Ψ/N` bytes, a barrier per step, a "
      f"real copy at each hop, and the byte counter incrementing where a chunk crosses from "
      f"rank `r` to rank `r+1`. Four checks before any of it is used:")
    a("")
    a("| check | result |")
    a("|---|---|")
    a(f"| ring all-reduce vs a naive sum over 32 ranks | agrees to **{r['ar_digits']:.1f} "
      f"digits** (`{r['ar_max_err']:.1e}` absolute) |")
    a(f"| reduce-scatter, reassembled | `{r['rs_max_err']:.1e}` |")
    a(f"| bytes per rank, measured at the hop | **{r['ar_bytes_per_rank_over_psi']:.4f}×Ψ** |")
    a(f"| bytes per rank, the ring's own theory `2Ψ(N−1)/N` | "
      f"**{r['ar_theory_over_psi']:.4f}×Ψ** |")
    a(f"| hops, measured / `N·2(N−1)` | {r['ar_hops']:,} / {r['ar_hops_theory']:,} |")
    a("")
    a(f"**An all-reduce does not move `2Ψ`.** It moves `{r['ar_bytes_per_rank_over_psi']:.4f}Ψ`, "
      f"and the missing `Ψ/{world}` is the one chunk you never have to send to yourself. It "
      f"is a small correction and I would not have found it by reading the formula.")
    a("")
    a("")
    a(f"(One 32-way all-reduce takes the simulator itself "
      f"{r['sim_wall_allreduce_ms']:.0f} ms of real Python threads. That number appears "
      f"nowhere else in this write-up and is used for nothing: it is a fact about 32 threads "
      f"on {r['host_cores']} cores, not about a GPU. Seconds, everywhere below, are "
      f"arithmetic over measured bytes and a stated bandwidth.)")
    a("")
    a(f"**And a flat ring across 4 nodes contains {r['flat_ring_inter_links']} inter-node "
      f"hops.** A ring is only as fast as its slowest hop, so a flat 32-way collective runs "
      f"at InfiniBand speed *end to end* even though {world - r['flat_ring_inter_links']} of "
      f"its {world} links are NVLink. §8 measures what that costs and what fixing it buys.")

    # ------------------------------------------------------- 2. the model
    cfg = r["cfg"]
    br = r["bytes_breakdown"]
    a("")
    a("## 2. The demo model, and the 16 bytes that start the whole problem")
    a("")
    a(f"A {cfg['n_layer']}-block GPT on character-level Shakespeare "
      f"({r['corpus_chars']:,} characters, {r['vocab']} distinct): `d_model = {cfg['d']}`, "
      f"{cfg['n_head']} heads, context {cfg['block']}, **Ψ = {psi:,} parameters** in "
      f"{r['n_tensors']} tensors and {r['n_groups']} sharding groups.")
    a("")
    a("It is written **functionally** — `gpt_forward(idx, P)` is handed a dict of parameter "
      "tensors rather than owning them — and that is not a style choice. ZeRO-3 does not "
      "have parameters, it has *shards*, and it rebuilds a layer's weights moments before "
      "using them and throws them away after. A model that owns `nn.Parameter`s cannot "
      "express that; a model that is handed its weights can, and then the *same forward "
      "function* runs under all four strategies — which is what makes §4's equivalence check "
      "mean anything. When DDP and ZeRO-3 disagree, it is the strategy disagreeing, not two "
      "different models.")
    a("")
    a("### Where 16 bytes per parameter comes from")
    a("")
    a("Every parameter in a mixed-precision Adam run is stored **five times**:")
    a("")
    a("| copy | dtype | bytes | what it is for |")
    a("|---|---|---|---|")
    a(f"| working weight | bf16 | {br['w16']} | what the matmuls read |")
    a(f"| gradient | bf16 | {br['g16']} | what backward writes |")
    a(f"| master weight | fp32 | {br['master']} | the *real* weight |")
    a(f"| Adam `m` | fp32 | {br['m']} | first moment |")
    a(f"| Adam `v` | fp32 | {br['v']} | second moment |")
    a(f"| | | **{r['bytes_per_param']}** | |")
    a("")
    a("The master copy is the one people forget, and it is the one that makes the arithmetic "
      "work. bf16 has an 8-bit mantissa — about three decimal digits — so `0.35 + 1e-4` "
      "rounds straight back to `0.35`. Apply the update in bf16 and the run looks like it is "
      "training while the weights do not move. Those 4 bytes are not redundancy; they are "
      "the only reason small steps accumulate.")
    a("")
    a(f"Asked afterwards what it had actually charged, the ledger reports "
      f"**{r['measured_bytes_per_param']:.4f} bytes per parameter** — the textbook 16, "
      f"arrived at by allocating five real tensors per weight and adding up their sizes.")
    a("")
    a(f"Sixteen bytes × {psi:,} parameters × **{world} GPUs** = "
      f"{mib(psi * r['bytes_per_param'] * world):.1f} MiB, of which "
      f"{mib(psi * r['bytes_per_param'] * (world - 1)):.1f} MiB is the same numbers again. "
      f"That is the problem ZeRO exists to solve, and it is the reason plain data "
      f"parallelism spends "
      f"{psi * r['bytes_per_param'] / (r['gpu_capacity_mib'] * MIB):.0%} of this card before "
      f"a single activation exists.")

    # ------------------------------------------------------- 3. the ledgers
    a("")
    a("## 3. What the 32 ledgers say")
    a("")
    a(f"One step, all four stages, same model, same data, same 32 GPUs, micro-batch "
      f"{r['micro_bsz']} (global batch {r['global_batch']}). Everything here is read out of "
      f"the ledgers; the theory column is printed **next to** the measurement, not "
      f"substituted for it.")
    a("")
    a("| | measured state/GPU | DeepSpeed formula | err | measured peak | activations | "
      "gathered bucket | comm/rank | vs DP |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in STAGES:
        a(f"| **{NAMES[s]}** | {mib(r[f's{s}_state']):.3f} MiB | "
          f"{mib(r[f's{s}_state_theory']):.3f} MiB | "
          f"{(r[f's{s}_state'] - r[f's{s}_state_theory']) / r[f's{s}_state_theory']:+.3%} | "
          f"{mib(r[f's{s}_peak']):.3f} MiB | {mib(r[f's{s}_act']):.3f} MiB | "
          f"{mib(r[f's{s}_transient']):.3f} MiB | "
          f"{r[f's{s}_comm_psi']:.4f}×Ψ | **{r[f's{s}_saving']:.2f}×** |")
    a("")
    a("Two columns there do not add up to the peak, and should not: *activations* and "
      "*gathered bucket* are each category's own high-water mark, and they do not occur at "
      "the same instant. The figure below stacks the breakdown the ledger held **at** the "
      "moment of peak instead, which does add up.")
    a("")
    a(f"**The formulas are reproduced to {r['state_vs_theory_worst']:.3%}.** That is the "
      f"headline, and not because the formulas needed confirming — it is the evidence that "
      f"what is implemented here is actually ZeRO and not something that merely allocates "
      f"less. A simulation that gets `4Ψ + 12Ψ/N` right by construction proves nothing; one "
      f"that gets it right by *allocating tensors and adding up their bytes* is a different "
      f"claim.")
    a("")
    a("And read down the middle of the table rather than across it — one question, asked "
      "three times:")
    a("")
    a("| stage | the question | what gets sharded | measured/GPU |")
    a("|---|---|---|---|")
    bc = {s: r[f"s{s}_bycat"] for s in STAGES}
    a(f"| **DP** | — | nothing; all {world} hold all of it | "
      f"{mib(r['s0_state']):.3f} MiB |")
    a(f"| **ZeRO-1** | does every GPU need its own optimizer state? | `master`, `m`, `v` — "
      f"12 of the 16 bytes | {mib(r['s1_state']):.3f} MiB |")
    a(f"| **ZeRO-2** | does every GPU need the whole gradient? | + gradients | "
      f"{mib(r['s2_state']):.3f} MiB |")
    a(f"| **ZeRO-3** | does every GPU need the whole model? | + weights | "
      f"{mib(r['s3_state']):.3f} MiB |")
    a("")
    a(f"The optimizer is the fat one — **12 of the 16 bytes** — which is why ZeRO-1 alone "
      f"already recovers {r['s1_saving']:.2f}× and why it is the cheapest thing anyone can "
      f"do. Note ZeRO-2's gradient line: it stores "
      f"{mib(bc[2]['grads']):.3f} MiB at peak against ZeRO-3's "
      f"{mib(bc[3]['grads']):.3f} MiB, and the difference is the reduce-scatter bucket that "
      f"exists only while a group's gradients are in flight.")
    a("")
    a("### The other redundancy, and the only arithmetic ZeRO removes")
    a("")
    a("Memory is not the only thing 32 identical copies waste. Under plain data parallelism "
      "every rank runs the Adam update on **every** weight and all 32 get the same answer — "
      "which the session put as *\"seven of the eight are repeating the work\"*. The sharded "
      "stages each update their own slice instead. Counted, not assumed:")
    a("")
    a("| | weights each rank Adam-steps | of Ψ | total work vs one GPU |")
    a("|---|---|---|---|")
    for s in STAGES:
        w = r[f"s{s}_opt_work"]
        a(f"| **{NAMES[s]}** | {w:,.0f} | {w / psi:.1%} | {w * world / psi:.2f}× |")
    a("")
    a(f"Plain data parallelism does **{r['opt_work_redundancy']:.0f}× the optimizer "
      f"arithmetic** of the sharded stages and throws {1 - 1 / world:.1%} of it away. It is "
      f"a small share of a real step — the optimizer is a handful of element-wise ops against "
      f"a forward and backward pass of matmuls — but it is worth being precise about what it "
      f"means: **this is the only arithmetic ZeRO removes.** The forward and backward FLOPs "
      f"are identical at every stage, which is exactly why §4's loss curves are identical. "
      f"ZeRO is a memory and communication optimisation that happens to delete some "
      f"duplicated optimizer work; it is not a way to do less of the actual training.")
    a("")
    a("![memory](results/memory.png)")

    # ------------------------------------------------------- 4. equivalence
    f64 = r["eq_f64_digits"]
    mix = r["eq_mixed_digits"]
    a("")
    a("## 4. The check that makes the other numbers mean anything")
    a("")
    a("ZeRO's claim is not \"a bit less memory for a bit less accuracy\" — it is that the "
      "sharded run computes **the same update** as the unsharded one. Memory savings are "
      "easy to fake, because doing less work also uses less memory. So before any table "
      "above is worth quoting, the four stages have to be shown to be four implementations "
      "of one algorithm.")
    a("")
    a(f"The reference is a single GPU that shards nothing: it takes all {world} "
      f"micro-batches itself, accumulates them, and applies one Adam step. That is the run "
      f"being distributed, so it is the thing to be equal to. Run twice, because the two "
      f"runs answer different questions:")
    a("")
    a(f"| after {r['eq_steps']} steps | stage 0 | stage 1 | stage 2 | stage 3 | what it says |")
    a("|---|---|---|---|---|---|")
    a("| **float64** — is the *algorithm* the same? | "
      + " | ".join(f"{k(f64, s):.1f} digits" for s in STAGES)
      + " | rounding is pushed to `1e-16`, so any real difference in what is computed shows up |")
    a("| **bf16/fp32** — is the run you would *launch* the same? | "
      + " | ".join(f"{k(mix, s):.1f} digits" for s in STAGES)
      + " | now rounding is `1e-3`, and this is the honest expectation |")
    a("")
    a(f"**In float64 every stage reproduces one GPU to {r['eq_f64_worst_digits']:.1f} "
      f"digits** — worst absolute disagreement on any weight "
      f"`{max(k(r['eq_f64_abs'], s) for s in STAGES):.1e}`. Same algorithm, not an "
      f"approximation of it.")
    a("")
    a(f"In bf16 the same four land {r['eq_mixed_worst_digits']:.1f} digits from the fp64 "
      f"reference and **{r['eq_mixed_stage_weight_spread']:.1e} from each other** — about one "
      f"fp32 ulp at this weight scale. Nearly identical, and nearly for a reason worth "
      f"knowing: all four route the gradient through the *same ring reduce-scatter*, so they "
      f"sum it in the same order. DDP's extra all-gather moves numbers around; it does not "
      f"change any of them. What residue is left comes from where each stage rounds the "
      f"gradient to bf16 — stages 0–2 after the whole backward, stage 3 at each layer "
      f"boundary.")
    a("")
    a("The lesson I take from the two rows: **a bit-exactness test against DDP is the wrong "
      "test to write.** The gap between 14 digits and 2 is rounding, and it is the same "
      "rounding the unsharded run already had.")

    # ------------------------------------------------------- 5. comm
    a("")
    a("## 5. Communication: stages 1 and 2 are free")
    a("")
    a("The byte counter has been running throughout. Per rank, per step, in units of the "
      "model's own size, counted at the hop:")
    a("")
    a("| stage | what it runs | measured | theory |")
    a("|---|---|---|---|")
    a(f"| DP | one all-reduce of the gradients | **{r['s0_comm_psi']:.4f}×Ψ** | "
      f"`2Ψ(N−1)/N` |")
    a(f"| ZeRO-1 | reduce-scatter the gradients, all-gather the weights | "
      f"**{r['s1_comm_psi']:.4f}×Ψ** | the same |")
    a(f"| ZeRO-2 | the same two, one bucket at a time | **{r['s2_comm_psi']:.4f}×Ψ** | "
      f"the same |")
    a(f"| ZeRO-3 | and a second all-gather, because the weights have to come back for the "
      f"backward | **{r['s3_comm_psi']:.4f}×Ψ** | `3Ψ(N−1)/N` |")
    a("")
    a(f"**ZeRO-1 and ZeRO-2 move exactly what DP moves.** Not approximately — the identical "
      f"count. This is the identity from §1 cashed out: DP's all-reduce *was* a "
      f"reduce-scatter followed by an all-gather all along, and these two stages just do "
      f"useful work in between. Moving a run from stage 0 to stage 2 costs **nothing** and "
      f"returns **{r['s2_saving']:.2f}×** the memory. There is no trade-off to weigh, and I "
      f"expected there to be one.")
    a("")
    a(f"ZeRO-3 costs **{r['s3_comm_psi'] / r['s0_comm_psi']:.1f}× the traffic**. Not 1.5× the "
      f"time — that depends entirely on what the wire is, which is §8.")
    a("")
    a(f"One more number from the same counter, because it sets up §8: of every collective's "
      f"bytes, **{r['s0_ib_share']:.1%} cross a node "
      f"boundary** — the {r['flat_ring_inter_links']} inter-node hops out of {world} in a "
      f"flat ring. A flat ring's InfiniBand *volume* was never the problem.")
    a("")
    a("![communication](results/communication.png)")

    # ------------------------------------------------------- 6. zero-3 peak
    bucket = r["bucket_rows"]
    shapes = r["shape_rows"]
    a("")
    a("## 6. What ZeRO-3's peak memory is actually set by")
    a("")
    a("§3 has a column the `16Ψ/N` formula does not predict, and it turned out to be the "
      "largest single entry in ZeRO-3's budget: the **gathered buffer**. A rank holding "
      f"1/{world} of the weights cannot multiply anything by them. It has to rebuild a whole "
      "group's weights first, use them, and drop them. So at its moment of peak memory a "
      "ZeRO-3 rank holds its shard of *everything* plus a full copy of *one group*:")
    a("")
    a("```")
    a("peak  =  16Ψ/N  +  (bytes in the largest gathered group)  +  activations")
    a("            ^              ^")
    a("            |              +-- does NOT shrink when you add GPUs")
    a("            +-- shrinks with N, and is the only term the formula mentions")
    a("```")
    a("")
    a(f"At 32 GPUs on this model that middle term is **{r['s3_transient_over_state']:.2f}× "
      f"the first one** — {mib(r['s3_transient']):.3f} MiB gathered against "
      f"{mib(r['s3_state']):.3f} MiB of shards. The usual summary, *\"ZeRO-3 gives you N× the "
      f"memory\"*, is describing the term that stopped mattering. Two things change it, and "
      f"neither is \"buy more GPUs\".")
    a("")
    a("**First, the size of the unit you gather** — DeepSpeed's `stage3_prefetch_bucket_size`, "
      "FSDP's wrapping policy:")
    a("")
    a("| bucket | buckets | largest | gathered buffer | peak/GPU | collectives/step |")
    a("|---|---|---|---|---|---|")
    for row in bucket:
        a(f"| {row['label']} | {row['n_groups']:.0f} | {row['biggest']:,.0f} params | "
          f"{mib(row['transient']):.3f} MiB | **{mib(row['peak']):.3f} MiB** | "
          f"{row['collectives']:.0f} |")
    a("")
    a(f"Gathering the whole model at once costs **{r['bucket_peak_ratio']:.1f}× the peak** of "
      f"gathering one block at a time, for identical shards and an identical result. That is "
      f"a \"fully sharded\" run that pays all of ZeRO-3's traffic and keeps none of its "
      f"saving — from one configuration line.")
    a("")
    a("Going the *other* way has a floor, and it is worth being precise because it is easy "
      "to get wrong: sharding **per tensor** rather than per block does not reduce the peak "
      "at all. All twelve of a block's tensors must be resident simultaneously for the block "
      "to compute. The peak is set by the granularity at which you can **free** weights — "
      "the unit of computation — not the granularity at which you store them. Finer buckets "
      "buy more collectives and no memory.")
    a("")
    a("**Second, the shape of the model.** The largest group is one transformer block, and a "
      "block goes as `12·d²`. Hold Ψ roughly fixed and make the model deeper and narrower:")
    a("")
    a("| shape | Ψ | shard/GPU | gathered | peak/GPU | collectives/step |")
    a("|---|---|---|---|---|---|")
    for row in shapes:
        a(f"| {row['n_layer']:.0f} blocks × d={row['d']:.0f} | {row['psi']:,.0f} | "
          f"{mib(row['state']):.3f} MiB | {mib(row['transient']):.3f} MiB | "
          f"**{mib(row['peak']):.3f} MiB** | {row['collectives']:.0f} |")
    a("")
    a(f"**{r['shape_transient_drop']:.1f}× less gathered memory for "
      f"{r['shape_collective_rise']:.1f}× the collectives, at essentially the same parameter "
      f"count.** The architecture decision and the sharding decision are one decision — which "
      f"is what the session was pointing at with \"17 layers across 8 GPUs\", arriving here "
      f"through memory instead of scheduling.")

    # ------------------------------------------------------- 7. capacity
    cap = r["cap_rows"]
    bsz = r["bsz_rows"]
    a("")
    a("## 7. The experiment a formula cannot do: grow it until it breaks")
    a("")
    a("Everything so far describes a model that fits. The claim people actually care about "
      "is the other one, and it needs no formula: make the model wider until a rank runs out "
      "of memory, and report the last width that worked.")
    a("")
    a("| | widest model that trained | Ψ | peak/GPU | vs DP |")
    a("|---|---|---|---|---|")
    for row in cap:
        s = int(row["stage"])
        a(f"| **{NAMES[s]}** | d = {row['d']:.0f} | {row['psi']:,.0f} | "
          f"{mib(row['peak']):.3f} MiB | **{row['vs_dp']:.1f}×** |")
    a("")
    a(f"**ZeRO-3 trained a model {r['cap_s3_over_dp']:.1f}× the size** of the largest one "
      f"plain data parallelism could hold, on identical hardware. That is the honest number "
      f"and it is *not* {world}×, for exactly the reason §6 gives: the gathered bucket does "
      f"not shard, and by the top of ZeRO-3's range it is most of the budget. Anyone quoting "
      f"{world}× is quoting the formula, not a run.")
    a("")
    a("And the ceiling is not decorative — the same width ZeRO-3 had just trained, attempted "
      "under plain data parallelism:")
    a("")
    a("```")
    a(f"DeviceOOM: {r['oom_demo_msg']}")
    a("```")
    a("")
    a("### The thing ZeRO does not shard")
    a("")
    a("Every stage above shards *state*. None of them touches **activations** — the tensors "
      "the backward pass needs, which scale with batch and sequence length and not at all "
      "with the number of GPUs. §3 already showed stages 0, 1 and 2 carrying an identical "
      f"{mib(r['s0_act']):.3f} MiB of them. So: at a fixed model, how large a micro-batch "
      "does each stage survive?")
    a("")
    a("| | largest micro-batch | peak/GPU | of which activations |")
    a("|---|---|---|---|")
    for row in bsz:
        s = int(row["stage"])
        a(f"| **{NAMES[s]}** | {row['bsz']:.0f} | {mib(row['peak']):.3f} MiB | "
          f"{mib(row['act']):.3f} MiB ({row['act'] / row['peak']:.0%}) |")
    a("")
    a(f"**Part of that last row is my implementation, not ZeRO**, and it is the one number "
      f"here most likely to mislead: my ZeRO-3 re-gathers weights in the backward pass via "
      f"`torch.utils.checkpoint`, which also discards activations and recomputes them. Real "
      f"ZeRO-3 re-gathers the weights *without* recomputing, so its activation footprint "
      f"would be the same {mib(r['s0_act']):.3f} MiB as the other three at batch 1, and its "
      f"batch limit correspondingly lower. The comparison that is clean is stages 0 → 1 → 2, "
      f"which share an identical forward: **{bsz[0]['bsz']:.0f} → {bsz[1]['bsz']:.0f} → "
      f"{bsz[2]['bsz']:.0f}**, bought purely with the state ZeRO freed up.")
    a("")
    a(f"ZeRO-3 runs **{bsz[3]['bsz'] / bsz[0]['bsz']:.0f}× the micro-batch** of plain data "
      f"parallelism — but at its own limit {bsz[3]['act'] / bsz[3]['peak']:.0%} of the card "
      f"is activations, which no ZeRO stage can do anything about. That is what gradient "
      f"checkpointing and sequence parallelism are for, and they are a different axis from "
      f"this one. It also means **the order of operations matters**: past a certain point, "
      f"more ZeRO buys nothing and the next win is somewhere else entirely.")

    # ------------------------------------------------------- 8. time
    a("")
    a("## 8. What the traffic costs in seconds — and why a faster GPU made it worse")
    a("")
    a(f"First, the flat ring's {r['flat_ring_inter_links']} inter-node hops from §1, and what "
      f"fixing them is worth. The same all-reduce, on the same 4 nodes, run two ways:")
    a("")
    a("| | InfiniBand traffic | modelled time |")
    a("|---|---|---|")
    a(f"| flat 32-ring | {mib(r['flat_ib_bytes']):.3f} MiB | {r['flat_model_ms']:.4f} ms |")
    a(f"| hierarchical (NVLink inside the box, IB only between node rails, on 1/8 of the "
      f"data) | {mib(r['hier_ib_bytes']):.3f} MiB | {r['hier_model_ms']:.4f} ms |")
    a("")
    a(f"**{r['hier_ib_reduction']:.1f}× less traffic on the slow wire and "
      f"{r['hier_speedup']:.1f}× faster**, for an identical result (the two losses agree to "
      f"`{r['hier_loss_match']:.1e}`).")
    a("")
    a(f"Those two ratios being so different is the whole lesson, and it is the thing I would "
      f"have got wrong by reasoning about bandwidth alone. Only "
      f"{r['flat_ib_share']:.1%} of a flat ring's bytes ever cross a node boundary, so its "
      f"InfiniBand *volume* is already small — cutting it by {r['hier_ib_reduction']:.1f}× is "
      f"not much of a prize. What the flat ring costs you is **time**: a ring step ends when "
      f"its slowest hop ends, so all `2(N−1)` of them run at InfiniBand speed even though 28 "
      f"links in 32 are NVLink. Restructuring the collective is worth "
      f"{r['hier_speedup']:.1f}×, far more than the byte count suggests.")
    a("")
    a(f"This is also where I got the bug worth confessing, and a memory table would never "
      f"have caught it: my first hierarchical implementation had only the *node "
      f"leaders* talk across nodes, which reduces one chunk in eight and silently leaves the "
      f"other seven node-local. It produced beautiful bandwidth numbers and the wrong "
      f"gradient. The cross-node groups have to be **rails** — all the local-rank-*j* GPUs, "
      f"one per node, all eight rails running at once. That is a test in "
      f"[`tests/`](tests/test_zero_harness.py) now.")
    a("")
    a("### From a toy to a machine that costs money")
    a("")
    a(f"The toy's step takes microseconds, so its wall-clock says nothing about a real run. "
      f"What carries over is the dimensionless thing the byte counter measured — **bytes "
      f"moved per parameter per step**, a property of the algorithm and not of the model. "
      f"Multiply by a real parameter count, divide by a real link, and the session's own "
      f"arithmetic falls out. Compute is modelled as `6·Ψ·tokens` FLOPs at an *achieved* "
      f"rate, not a peak one: "
      + ", ".join(f"{name} at {f / 1e12:.0f} TFLOP/s"
                  for name, f in r["proj_hardware"].items()) + ".")
    a("")
    a(f"A **{r['proj_psi'] / 1e9:.0f}B-parameter** model, {r['proj_tokens']:,.0f} tokens per "
      f"GPU per step, no overlap (the upper bound):")
    a("")
    a("| | B/param | GPU | compute | comm on NVLink | idle | comm on InfiniBand | idle |")
    a("|---|---|---|---|---|---|---|---|")
    for s in STAGES:
        for hw in r["proj_hardware"]:
            rows = [p for p in r["proj_rows"] if int(p["stage"]) == s and p["hw"] == hw]
            nv = [p for p in rows if "NVLink" in p["link"]][0]
            ib = [p for p in rows if "Infini" in p["link"]][0]
            label = f"**{NAMES[s]}**" if hw == list(r["proj_hardware"])[0] else ""
            a(f"| {label} | {k(r['proj_bytes_per_param'], s):.3f} | {hw} | "
              f"{nv['compute_s']:.3f} s | {nv['comm_s']:.3f} s | {nv['bubble']:.1%} | "
              f"{ib['comm_s']:.3f} s | **{ib['bubble']:.1%}** |")
    a("")
    a(f"**The session's point, reproduced.** The same communication is "
      f"{r['bubble_dp_h100_ib']:.0%} of an H100 step and {r['bubble_dp_b200_ib']:.0%} of a "
      f"B200 step. Buying the faster GPU made the idle fraction **worse**, because it "
      f"shortened the only half of the step that got faster. The bytes do not care how fast "
      f"your tensor cores are.")
    a("")
    a(f"And the answer to *\"what does ZeRO-3 cost\"*: inside one node, its 1.5× traffic is "
      f"{r['bubble_s3_h100_nv']:.1%} of the step — almost nothing. Across InfiniBand it is "
      f"most of your money. **ZeRO-3 is a decision about the wire, not about memory.**")

    # ------------------------------------------------------- 9. scaling
    sc = r["scale_rows"]
    worlds = [row["world"] for row in k(sc, 0)]
    a("")
    a("## 9. Adding GPUs: what each stage does with them")
    a("")
    a("Sweeping the world size from 1 to 32 separates two things people conflate when they "
      "say \"ZeRO scales\". Every point is a real setup and a real step on a real world of "
      "that size.")
    a("")
    a("| state/GPU | " + " | ".join(f"N={w:.0f}" for w in worlds) + " |")
    a("|---" * (len(worlds) + 1) + "|")
    for s in STAGES:
        a(f"| **{NAMES[s]}** | "
          + " | ".join(f"{mib(row['state']):.3f} MiB" for row in k(sc, s)) + " |")
    a("")
    a(f"Stage 0's memory is **flat in N** — it changed by "
      f"{r['scale_s0_flat']:.2f}× from 1 GPU to 32, because it cannot change. The 32nd GPU "
      f"adds throughput and not one byte of capacity. The sharded stages fall as `a + b/N`, "
      f"and the interesting part is `a`, the piece that does not shard: **`4Ψ` for ZeRO-1, "
      f"`2Ψ` for ZeRO-2, nothing for ZeRO-3.**")
    a("")
    a(f"Which is why ZeRO-1 is already **{r['scale_s1_floor']:.0%} of the way to its floor** "
      f"at N=32: doubling to 64 GPUs would buy it almost nothing. Only stage 3 keeps paying "
      f"as the cluster grows, and that — not the size of the saving at any one N — is the "
      f"actual argument for it.")

    # ------------------------------------------------------- 10. session table
    sess = r["session_rows"]
    a("")
    a("## 10. Back to the session's own numbers")
    a("")
    a(f"Everything here runs at 1/{r['scale_to_80gib']:,} scale, so the last thing worth "
      f"doing is pushing a measurement back up and checking it against numbers that were not "
      f"produced here: the session's table for a 30-billion-parameter model on 8 GPUs.")
    a("")
    a("Nothing below evaluates the published formula. Every stage's per-GPU state has the "
      "form `a + b/N` bytes per parameter — `a` the part that never shards, `b` the part "
      "that does — and **both coefficients are solved for out of §9's measured sweep**. Two "
      "world sizes determine them; the four in between are a check on the fit, not an input "
      "to it.")
    a("")
    a("| | fitted `a` | fitted `b` | what the ZeRO paper says | worst measured point |")
    a("|---|---|---|---|---|")
    textbook = {0: "(16, 0)", 1: "(4, 12)", 2: "(2, 14)", 3: "(0, 16)"}
    for s in STAGES:
        f = k(r["fit_ab"], s)
        a(f"| **{NAMES[s]}** | {f['a']:.4f} B | {f['b']:.4f} B | {textbook[s]} | "
          f"{f['resid']:.1e} B |")
    a("")
    a(f"The fit reproduces every measured world size to **{r['fit_worst_resid']:.1e} bytes "
      f"per parameter** and the published coefficients to **{r['fit_vs_textbook']:.1e}**. "
      f"So the simulator is not merely producing plausible numbers — it recovers ZeRO's own "
      f"constants from allocations it actually made.")
    a("")
    a("Pushed out to 30B parameters on 8 GPUs:")
    a("")
    a("| | extrapolated from this notebook | the session said | difference |")
    a("|---|---|---|---|")
    for row in r["session_rows"]:
        s = int(row["stage"])
        a(f"| **{NAMES[s]}** | {row['gib']:.1f} GiB | {row['session']:.1f} GiB | "
          f"{row['diff']:+.2%} |")
    a("")
    a(f"Worst disagreement **{r['session_worst_diff']:.2%}**, which also says the "
      f"1/{r['scale_to_80gib']:,} scaling did not distort anything that matters. And the "
      f"thing the table is really saying: on an 80 GiB card a 30B model under plain data "
      f"parallelism needs {r['session_rows'][0]['gib']:.0f} GiB per GPU, so **it does not "
      f"start**. Under ZeRO-3 on 8 GPUs it needs {r['session_rows'][3]['gib']:.1f}, so it "
      f"does — with ~{80 - r['session_rows'][3]['gib']:.0f} GiB left over, which is the "
      f"number that actually decides the batch size, and which §7 says is the next thing to "
      f"run out.")
    a("")

    # ------------------------------------------------------- 11. verdict
    a("")
    a("## 11. Which stage, and when")
    a("")
    a("| stage | take it when | the cost | the catch |")
    a("|---|---|---|---|")
    a(f"| **ZeRO-1** | always | **zero extra bytes** | floors at `4Ψ`; past ~16 GPUs more "
      f"ranks buy almost nothing |")
    a(f"| **ZeRO-2** | always — strictly better than ZeRO-1 | **zero extra bytes** | needs "
      f"the gradient reduced *during* the backward, so a framework that reduces afterwards "
      f"is really ZeRO-1 wearing a hat; floors at `2Ψ` |")
    a(f"| **ZeRO-3** | when the model does not otherwise fit, **and** the GPUs are in one box "
      f"| {r['s3_comm_psi'] / r['s0_comm_psi']:.1f}× the traffic: "
      f"{r['bubble_s3_h100_nv']:.1%} of an H100 step on NVLink, "
      f"{[p for p in r['proj_rows'] if int(p['stage']) == 3 and p['hw'] == 'H100' and 'Infini' in p['link']][0]['bubble']:.0%} "
      f"across InfiniBand | the peak is set by the largest gathered bucket, not by "
      f"`16Ψ/N`; check the bucket and the block width first |")
    a("")
    a("Three of these I would have got right from the lecture. The one I would not is the "
      "middle column being **zero** for stages 1 and 2 — I went in expecting to trade "
      "bandwidth for memory and there is nothing to trade. And the ZeRO-3 row is the one "
      "that changed most on contact with a measurement: I would have written \"take ZeRO-3 "
      "when the model is big\", and the measurement says the model size is the *second* "
      "question. The first is how many boxes.")
    a("")
    a(f"*Produced by `zero_harness.ipynb` on {r['host_cores']} CPU cores, torch "
      f"{r['torch_version']}, seed {r['seed']}, in {r['elapsed_s'] / 60:.1f} minutes. All "
      f"{r['n_results_keys']} recorded values are read out of "
      f"[`results/results.json`](results/results.json) by `tools/render_numbers.py`, and the "
      f"log, the figures, the JSON and every table above are regenerated together from one "
      f"execution. The memory and byte counts are integers and should reproduce exactly on "
      f"any machine; the modelled seconds are arithmetic over those integers and the stated "
      f"bandwidths, and the simulator's own wall-clock is not used for anything.*")
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
