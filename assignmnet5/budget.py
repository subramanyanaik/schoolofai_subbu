"""
budget.py — Derive the token budget from compute instead of asserting it.

The single most important correction from the earlier 40B draft:
for a Mixture-of-Experts model the training FLOPs scale with ACTIVE
parameters, not total parameters.

    C  =  6 * N_active * D          (fwd+bwd, dense-equivalent)

A 120B MoE with 15B active costs the same per token as a 15B dense model,
but carries the knowledge capacity of something far larger. That is exactly
why V4 could train 120B on a small node -- and it is why the token budget
must be computed from N_active, while the DATA DIVERSITY requirement is
driven by N_total (more experts = more distinct things to specialise in).
"""
from . import config as C


def cluster_flops(cl=None):
    """Total training FLOPs available from the cluster."""
    cl = cl or C.CLUSTER
    per_gpu = cl["gpu_peak_tflops"] * 1e12 * cl["mfu"] * cl["precision_speedup"]
    seconds = cl["days"] * 24 * 3600
    return per_gpu * cl["n_gpus"] * seconds


def tokens_from_flops(flops, n_active):
    """Invert C = 6*N*D."""
    return flops / (6.0 * n_active)


def flops_for_run(n_active, tokens):
    return 6.0 * n_active * tokens


def chinchilla_floor(n_active):
    return C.CHINCHILLA_TOK_PER_PARAM * n_active


def overtrain_ratio(n_active, tokens):
    return tokens / n_active


def budget_report():
    m = C.MODELS[C.FLAGSHIP]
    n_act, n_tot = m["active"], m["total"]
    avail = cluster_flops()
    feasible = tokens_from_flops(avail, n_act)
    planned = C.TRAIN_BUDGET
    needed = flops_for_run(n_act, planned)

    # V4 back-reference: what did V4 actually cost?
    v4_flops_if_15b_active = flops_for_run(15e9, C.V4["tokens"])

    return {
        "model": C.FLAGSHIP,
        "total_params": n_tot,
        "active_params": n_act,
        "sparsity": n_act / n_tot,
        "cluster_flops": avail,
        "feasible_tokens": feasible,
        "planned_tokens": planned,
        "flops_needed": needed,
        "compute_headroom": avail / needed,
        "chinchilla_floor": chinchilla_floor(n_act),
        "overtrain_x_chinchilla": planned / chinchilla_floor(n_act),
        "tok_per_active_param": overtrain_ratio(n_act, planned),
        "tok_per_total_param": planned / n_tot,
        "v4_tokens": C.V4["tokens"],
        "v5_over_v4_tokens": planned / C.V4["tokens"],
        "v4_flops_ref": v4_flops_if_15b_active,
        "flops_vs_v4": needed / v4_flops_if_15b_active,
        "collection_target": C.COLLECTION_TARGET,
        "selection_headroom": C.COLLECTION_TARGET / planned,
    }


def variant_table():
    """Token budget each family member would need at the same overtrain ratio."""
    rows = []
    for name, m in C.MODELS.items():
        na = m["active"]
        # proxies run at Chinchilla floor (cheap); production models overtrain
        ratio = C.CHINCHILLA_TOK_PER_PARAM if "Proxy" in name else C.OVERTRAIN_TARGET
        toks = ratio * na
        rows.append(dict(
            name=name, kind=m["kind"], total=m["total"], active=na,
            role=m["role"], tok_per_active=ratio, tokens=toks,
            flops=flops_for_run(na, toks),
        ))
    return rows
