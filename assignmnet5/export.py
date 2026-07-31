"""
export.py — dump every computed number as JSON.

Purpose: a reviewer should be able to check a claim in the README against
machine-readable output without reading the code or trusting a table that
was typed by hand. Run:

    python3 -m erav5.export > ../results/plan.json
"""
import json
from . import config as C
from . import budget as B
from . import mixture as M
from . import curriculum as CU
from . import floors as F
from . import proxy as P
from . import sensitivity as S
from . import validate as V


def build():
    rows = M.solve()
    checks = V.run_all(rows)
    return {
        "meta": {
            "flagship": C.FLAGSHIP,
            "train_budget_tokens": C.TRAIN_BUDGET,
            "collection_target_tokens": C.COLLECTION_TARGET,
            "vocab_size": C.VOCAB_SIZE,
        },
        "budget": B.budget_report(),
        "model_family": B.variant_table(),
        "mixture": rows,
        "totals": M.totals(rows),
        "indic": M.indic_rollup(rows),
        "starved": [r["lane"] for r in M.starved_lanes(rows)],
        "stages": CU.stage_tokens(),
        "stage_mix": CU.STAGE_MIX,
        "integration": CU.integrate(),
        "blend_bands": CU.blend_bands(),
        "bands": CU.BANDS,
        "effort_levels": CU.EFFORT_LEVELS,
        "long_context_pool": CU.LONG_CTX_POOL,
        "floor": F.floor_report(rows),
        "reserve": F.reserve_report(),
        "verified_native_exposure": F.verified_native_exposure(rows),
        "fertility": {
            "weighted_indic": F.weighted_indic_fertility(),
            "effective_words": F.effective_words(M.indic_rollup(rows)["total"]),
            "session2_balance": F.session2_balance_score(),
        },
        "proxies": P.proxy_table(),
        "proxy_total": P.proxy_total(),
        "sensitivity": S.sensitivity_table(),
        "cleaning_plan": S.cleaning_plan(rows),
        "cleaning_priority": S.priority_queue(rows),
        "validation": {
            "total": len(checks.rows),
            "passed": len(checks.rows) - len(checks.failed),
            "failed": [{"check": n, "detail": d} for n, ok, d in checks.rows if not ok],
            "all": [{"check": n, "pass": ok, "detail": d} for n, ok, d in checks.rows],
        },
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, default=str))
