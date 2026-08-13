"""Parameter audit -- the number this whole project is about.

Trap 5 made this a script rather than a comment.  Assigning an ``nn.Module``
somewhere just to borrow its buffers silently registers that module's unused
weights as parameters of the host.  They get no gradient, AdamW skips them, so
training is completely unaffected -- and only the *reported parameter count* is
wrong.  ``unused_params`` must be empty for every arm.
"""

from __future__ import annotations

import torch

from _common import rule, table, write_json
from kronecker_v2.model import ARMS
from kronecker_v2.train import build_arm
from kronecker_v2.vocab import gpt2_token_bytes

D_C, L_MAX = 32, 128


def main() -> None:
    toks = gpt2_token_bytes()
    rows, audit = [], {}
    for arm in ARMS:
        model, packed, codec = build_arm(arm, toks, d_c=D_C, L_max=L_MAX, seed=0)
        bd = model.param_breakdown()
        unused = model.unused_parameters()
        head_is_vocab_free = bd["head"] == 0 or not any(
            p.numel() >= packed.V for n, p in model.lm_head.named_parameters()
            if n != "unigram"
        )
        rec = {
            **bd,
            "arm": arm,
            "unused_params": unused,
            "vocab": packed.V,
            "codec": repr(codec) if codec else None,
            "head_has_no_dense_vocab_matrix": bool(head_is_vocab_free),
        }
        audit[arm] = rec
        rows.append([
            arm, f"{bd['total'] / 1e6:.2f}M", f"{bd['body'] / 1e6:.2f}M",
            f"{bd['input_embedding'] / 1e6:.2f}M", f"{bd['head'] / 1e6:.2f}M",
            "OK" if not unused else ",".join(unused),
        ])
        del model
    table(rows, ["arm", "total", "body", "input", "head", "unused"])

    rule("vocabulary independence")
    inv_rows = []
    for arm in ARMS:
        counts = []
        for V in (8192, 32768):
            sub = toks[:V]
            m, _, _ = build_arm(arm, sub, d_c=D_C, L_max=L_MAX, seed=0)
            counts.append(m.n_params())
            del m
        delta = counts[1] - counts[0]
        inv_rows.append([arm, f"{counts[0]:,}", f"{counts[1]:,}", f"{delta:+,}"])
        audit[arm]["delta_params_8k_to_32k_vocab"] = delta
    table(inv_rows, ["arm", "V=8,192", "V=32,768", "delta"])

    gates = {
        "no_unused_params": all(not v["unused_params"] for v in audit.values()),
        "a2_vocab_independent": audit["a2_kronf_kas"]["delta_params_8k_to_32k_vocab"] == 0,
        "a3_vocab_independent": audit["a3_kronf_kas_tied"]["delta_params_8k_to_32k_vocab"] == 0,
        "a4_costs_one_scalar_per_token":
            audit["a4_kas_unigram"]["delta_params_8k_to_32k_vocab"] == 32768 - 8192,
        "a0_costs_d_model_per_token":
            audit["a0_bpe_tied"]["delta_params_8k_to_32k_vocab"] == (32768 - 8192) * 384,
    }
    rule("gates")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    write_json("params_audit.json", {"arms": audit, "gates": gates,
                                     "all_gates_pass": all(gates.values())})
    if not all(gates.values()):
        raise SystemExit("params audit gate failed")


if __name__ == "__main__":
    main()
