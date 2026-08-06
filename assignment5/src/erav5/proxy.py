"""
proxy.py — the mixture as a testable hypothesis.

Every proxy is specified as: hypothesis -> arms -> metric -> decision rule,
with the cost computed so a reviewer can see it is actually affordable.
A decision rule that cannot fire in the negative direction is not a test,
so each rule states what would REFUTE the plan, not only what confirms it.
"""
from . import config as C
from .budget import flops_for_run

GPU_HOUR_USD = 2.50   # [EST] spot H100


def _cost(n_active, tokens, arms=2):
    fl = flops_for_run(n_active, tokens) * arms
    per_gpu = C.CLUSTER["gpu_peak_tflops"] * 1e12 * C.CLUSTER["mfu"]
    gpu_hours = fl / per_gpu / 3600
    return dict(flops=fl, gpu_hours=gpu_hours, usd=gpu_hours * GPU_HOUR_USD)


PROXIES = [
    dict(
        id="P1", scale="3B", n_active=3e9, tokens=60e9, arms=2,
        title="Indic synthetic parity cap",
        hypothesis=("Capping synthetic Indic at <=1.0x native beats a "
                    "synthetic-heavy arm on native-quality metrics at matched compute."),
        arm_a="plan mixture (synthetic at the computed synthetic/native ratio)",
        arm_b="synthetic DOUBLED (2x the plan ratio), translated held constant",
        metric="MILU per Tier-1 lang; FLORES chrF++; 200-sample native audit",
        rule=("Adopt the cap only if A beats B by >=2.0 pts mean MILU AND shows no "
              "chrF++ regression. If B wins, the cap RISES -- the plan was too "
              "conservative. If neither separates, drop the cap as unjustified machinery."),
    ),
    dict(
        id="P2", scale="1B", n_active=1e9, tokens=20e9, arms=2,
        title="Protected floor necessity",
        hypothesis=("Without a native-only always-on floor, Indic capability drifts "
                    "down during the code/math-heavy S2 phase."),
        arm_a="8% native-Indic floor enforced every batch",
        arm_b="same global Indic share, no batch-level floor (selector free)",
        metric=("MILU (hi + one Dravidian) at 5 checkpoints -- the CURVE SHAPE is the "
                "evidence. Secondary: HumanEval/GSM-mini to confirm no tax on dominant lanes."),
        rule=("Keep the floor if B shows a mid-training MILU dip >=3 pts that A avoids. "
              "If both curves are flat, DROP the floor -- it is unnecessary complexity."),
    ),
    dict(
        id="P3", scale="1B", n_active=1e9, tokens=20e9, arms=2,
        title="Agentic verification gate",
        hypothesis=("Execution-verified sandbox trajectories beat imitation-only public "
                    "agent data at equal token count."),
        arm_a="sandbox self-play, kept only on task success",
        arm_b="public trajectories, no verification gate",
        metric="BFCL function-call accuracy; tau2-bench task success",
        rule=("A must clear B by >=5 pts on BOTH. If not, the generation pipeline is the "
              "defect to fix before the 6% agentic share scales -- not a reason to cut the lane."),
    ),
    dict(
        id="P4", scale="3B", n_active=3e9, tokens=60e9, arms=3,
        title="Mixture-transition gradient stability",
        hypothesis=("Ramping lane-share changes across a >=100B-token blending band "
                    "prevents the ~150x gradient-norm spike V4 recorded on an abrupt "
                    "Hindi share increase."),
        arm_a="100B-token linear ramp between stage mixes",
        arm_b="abrupt step change at the stage boundary",
        arm_c="20B-token ramp (is the band over-specified?)",
        metric="max gradient-norm multiplier over the transition; loss-spike count",
        rule=("Keep the 100B band only if the abrupt arm spikes >=5x AND the 20B arm "
              "also spikes. If 20B is sufficient, shorten the band and reclaim schedule."),
    ),
    dict(
        id="P5", scale="1B", n_active=1e9, tokens=20e9, arms=2,
        title="Loss-masking on tool observations",
        hypothesis=("Zero-loss masking of tool outputs prevents observation "
                    "hallucination without costing task success."),
        arm_a="loss on model-generated tokens only (plans, code, JSON calls)",
        arm_b="loss on the full trajectory including tool responses",
        metric="rate of fabricated tool observations in held-out rollouts; BFCL",
        rule=("A must cut fabricated-observation rate by >=50% at no BFCL cost. This is "
              "the cheapest proxy in the set and should run FIRST."),
    ),
]


def proxy_table():
    rows = []
    for p in PROXIES:
        c = _cost(p["n_active"], p["tokens"], p["arms"])
        rows.append(dict(**p, **c))
    return rows


def proxy_total():
    rows = proxy_table()
    tot_usd = sum(r["usd"] for r in rows)
    flag_flops = flops_for_run(C.MODELS[C.FLAGSHIP]["active"], C.TRAIN_BUDGET)
    per_gpu = C.CLUSTER["gpu_peak_tflops"] * 1e12 * C.CLUSTER["mfu"]
    flag_usd = flag_flops / per_gpu / 3600 * GPU_HOUR_USD
    return dict(total_usd=tot_usd, flagship_usd=flag_usd,
                pct_of_flagship=tot_usd / flag_usd,
                total_gpu_hours=sum(r["gpu_hours"] for r in rows))
