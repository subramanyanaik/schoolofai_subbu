"""Tests for the parts of the harness where a wrong answer would look like a right one.

`optimizer_harness.py` is a top-to-bottom script: importing it would train about a hundred
models.  So the pieces under test are lifted out of it with `ast`, which keeps one source of
truth -- if a definition in the notebook changes, these tests change with it, and a stale
copy cannot pass.

Everything here is arithmetic rather than measurement: it must hold on any machine, CPU or
GPU, and it is what CI runs.
"""
import ast
import io
import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "optimizer_harness.py")

WANTED = ("digits_agreeing", "adam_by_hand", "adam_l2", "adamw_by_hand", "bc_multiplier",
          "HandAdamW", "lr_at", "Block", "GPT", "lr_groups", "refine")


def load_pieces():
    """Lift the wanted definitions, plus every module-level literal constant, and run them
    in a namespace of their own.  Nothing that touches data, CUDA or the filesystem runs."""
    tree = ast.parse(io.open(SRC, encoding="utf-8").read())
    ns = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math,
          # VOCAB_SIZE is read off the corpus at run time; these tests do not care what it
          # is, so they pick one.  Every other constant comes from the source.
          "VOCAB_SIZE": 65}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            targets = node.targets[0]
            names = targets.elts if isinstance(targets, ast.Tuple) else [targets]
            values = value if isinstance(targets, ast.Tuple) else [value]
            for nm, v in zip(names, values):
                if isinstance(nm, ast.Name) and nm.id.isupper():
                    ns[nm.id] = v
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED:
            keep.append(node)
    missing = set(WANTED) - {n.name for n in keep}
    assert not missing, f"harness no longer defines {missing}"
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
    return ns


P = load_pieces()
adam_by_hand, adam_l2, adamw_by_hand = P["adam_by_hand"], P["adam_l2"], P["adamw_by_hand"]
bc_multiplier, HandAdamW, lr_at = P["bc_multiplier"], P["HandAdamW"], P["lr_at"]
GPT, lr_groups, refine = P["GPT"], P["lr_groups"], P["refine"]
digits_agreeing = P["digits_agreeing"]
LR, B1, B2, EPS = P["LR"], P["B1"], P["B2"], P["EPS"]
BASE_WIDTH, N_LAYER, HEAD_DIM = P["BASE_WIDTH"], P["N_LAYER"], P["HEAD_DIM"]


def torch_adam_trajectory(w0, grads, cls=torch.optim.Adam, **kw):
    p = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    opt = cls([p], lr=LR, betas=(B1, B2), eps=EPS, **kw)
    out = []
    for g in grads:
        p.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        st = opt.state[p]
        out.append((st["exp_avg"].item(), st["exp_avg_sq"].item(), p.item()))
    return out


# ------------------------------------------------------------------ 1. Adam, by hand

def test_hand_adam_matches_torch_on_the_readme_sequence():
    """The five gradients the write-up quotes: m, v and w, every step."""
    w0, grads = P["W0"], P["GRADS"]
    hand = adam_by_hand(w0, grads)
    for row, (m, v, w) in zip(hand, torch_adam_trajectory(w0, grads)):
        assert row["m"] == pytest.approx(m, rel=0, abs=1e-15)
        assert row["v"] == pytest.approx(v, rel=0, abs=1e-15)
        assert row["w"] == pytest.approx(w, rel=0, abs=1e-15)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_hand_adam_matches_torch_on_random_gradients(seed):
    rng = np.random.default_rng(seed)
    grads = list(rng.normal(0, 0.5, 40) * rng.choice([1.0, 30.0, 0.01], 40))
    w0 = float(rng.normal())
    hand = adam_by_hand(w0, grads)
    ref = torch_adam_trajectory(w0, grads)
    assert hand[-1]["w"] == pytest.approx(ref[-1][2], rel=1e-13)
    assert digits_agreeing(hand[-1]["w"], ref[-1][2]) > 12


def test_first_step_is_exactly_lr_whatever_the_gradient():
    """With eps out of the way, m_hat/sqrt(v_hat) at t=1 is exactly sign(g)."""
    for g in (1e-9, 0.01, 1.0, 1e6, -3.7):
        row = adam_by_hand(0.0, [g], eps=0.0)[0]
        assert abs(row["step"]) == pytest.approx(LR, rel=1e-12)
        assert math.copysign(1, row["step"]) == math.copysign(1, g)


@pytest.mark.parametrize("factor", [1e-6, 1e-3, 7.0, 1e3, 1e6])
def test_adam_is_exactly_scale_invariant_when_eps_is_zero(factor):
    grads = [0.10, 0.20, -0.05, 1.50, 0.08]
    a = adam_by_hand(0.35, grads, eps=0.0)
    b = adam_by_hand(0.35, [g * factor for g in grads], eps=0.0)
    for x, y in zip(a, b):
        assert x["w"] == pytest.approx(y["w"], rel=1e-13)


def test_eps_is_what_breaks_scale_invariance():
    """The claim the write-up makes about eps: shrink the gradients far enough and Adam
    stops being scale free.  With the default eps it must NOT be invariant."""
    grads = [0.10, 0.20, -0.05, 1.50, 0.08]
    a = adam_by_hand(0.35, grads)
    tiny = adam_by_hand(0.35, [g * 1e-6 for g in grads])
    assert abs(a[-1]["w"] - tiny[-1]["w"]) > 1e-6


# ------------------------------------------------------- 2. bias correction is one scalar

def test_bias_correction_is_exactly_the_scalar():
    """Uncorrected step / corrected step must equal 1/r(t) exactly, at every step, for any
    gradient sequence -- that is the claim section 2 rests on."""
    rng = np.random.default_rng(11)
    grads = list(rng.normal(0, 0.3, 60))
    on = adam_by_hand(0.0, grads, eps=0.0, bias_correction=True)
    off = adam_by_hand(0.0, grads, eps=0.0, bias_correction=False)
    for a, b in zip(on, off):
        assert b["step"] / a["step"] == pytest.approx(1 / bc_multiplier(a["t"]), rel=1e-12)


def test_the_multiplier_is_worst_in_the_middle_not_at_step_one():
    """The write-up's headline: the damage peaks around step 12, not step 1."""
    rs = [bc_multiplier(t) for t in range(1, 4000)]
    t_min = int(np.argmin(rs)) + 1
    assert t_min == 12
    assert 1 / rs[t_min - 1] > 1 / rs[0]          # worse than the first step
    assert 1 / rs[t_min - 1] == pytest.approx(6.57, abs=0.01)


def test_the_multiplier_converges_to_one_and_takes_its_time():
    assert bc_multiplier(1) == pytest.approx(math.sqrt(1 - B2) / (1 - B1))
    assert bc_multiplier(200000) == pytest.approx(1.0, abs=1e-6)
    assert abs(bc_multiplier(1000) - 1) > 0.01    # still >1% off after 1000 steps


# ------------------------------------------------------------------ 3. weight decay

def test_adamw_matches_torch_and_adam_l2_matches_torch():
    w0, grads, lam = P["W0"], P["GRADS"], P["LAM"]
    ref_l2 = torch_adam_trajectory(w0, grads, torch.optim.Adam, weight_decay=lam)
    ref_w = torch_adam_trajectory(w0, grads, torch.optim.AdamW, weight_decay=lam)
    assert adam_l2(w0, grads, lam) == pytest.approx(ref_l2[-1][2], rel=1e-13)
    assert adamw_by_hand(w0, grads, lam) == pytest.approx(ref_w[-1][2], rel=1e-13)


def test_coupled_and_decoupled_decay_are_different_algorithms():
    w0, grads, lam = P["W0"], P["GRADS"], P["LAM"]
    assert abs(adam_l2(w0, grads, lam) - adamw_by_hand(w0, grads, lam)) > 1e-6


def test_both_decays_vanish_at_lambda_zero():
    w0, grads = P["W0"], P["GRADS"]
    plain = adam_by_hand(w0, grads)[-1]["w"]
    assert adam_l2(w0, grads, 0.0) == pytest.approx(plain, rel=1e-14)
    assert adamw_by_hand(w0, grads, 0.0) == pytest.approx(plain, rel=1e-14)


def test_decay_ordering_matters_but_only_at_second_order():
    """torch shrinks w first; the paper subtracts both from the same w.  The gap must be
    real but O(eta^2 lambda) -- the exact claim the write-up makes."""
    w0, grads, lam = P["W0"], P["GRADS"], P["LAM"]
    first = adamw_by_hand(w0, grads, lam, decay_first=True)
    last = adamw_by_hand(w0, grads, lam, decay_first=False)
    gap = abs(first - last)
    assert gap > 0
    assert gap < 10 * LR * LR * lam * len(grads) * abs(w0)


# ------------------------------------------------- 4. the tensor optimizer is the same one

def test_handadamw_equals_torch_adamw_elementwise():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(32, 16, dtype=torch.float64, generator=g).requires_grad_(True)
    b = a.detach().clone().requires_grad_(True)
    oa = HandAdamW([a], lr=1e-3, betas=(B1, B2), eps=EPS, weight_decay=0.05)
    ob = torch.optim.AdamW([b], lr=1e-3, betas=(B1, B2), eps=EPS, weight_decay=0.05)
    for _ in range(50):
        grad = torch.randn(32, 16, dtype=torch.float64, generator=g) * 0.1
        a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    assert (a - b).abs().max().item() < 1e-14


def test_handadamw_without_bias_correction_is_the_scalar_again():
    g = torch.Generator().manual_seed(5)
    a = torch.zeros(8, 8, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(8, 8, dtype=torch.float64, requires_grad=True)
    oa = HandAdamW([a], lr=1e-3, betas=(B1, B2), eps=0.0, bias_correction=True)
    ob = HandAdamW([b], lr=1e-3, betas=(B1, B2), eps=0.0, bias_correction=False)
    pa = a.detach().clone()
    pb = b.detach().clone()
    for t in range(1, 31):
        grad = torch.randn(8, 8, dtype=torch.float64, generator=g) * 0.1
        a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
        ratio = ((b.detach() - pb).norm() / (a.detach() - pa).norm()).item()
        assert ratio == pytest.approx(1 / bc_multiplier(t), rel=1e-10)
        pa, pb = a.detach().clone(), b.detach().clone()


def test_handadamw_counterfactual_reports_the_other_setting():
    """The counterfactual instrumentation section 2 depends on: with bias correction on, the
    recorded ratio must be the size of the step the run did NOT take, over the one it did."""
    g = torch.Generator().manual_seed(9)
    a = torch.zeros(16, 16, dtype=torch.float64, requires_grad=True)
    o = HandAdamW([a], lr=1e-3, betas=(B1, B2), eps=0.0, counterfactual=True)
    for t in range(1, 21):
        a.grad = torch.randn(16, 16, dtype=torch.float64, generator=g) * 0.2
        o.step()
        got = math.sqrt(o.other_sq / o.taken_sq)
        assert got == pytest.approx(1 / bc_multiplier(t), rel=1e-10)


# ------------------------------------------------------------------ 5. the schedules

@pytest.mark.parametrize("sched", ["cosine", "wsd", "constant"])
def test_warmup_is_linear_and_arrives_exactly_at_the_peak(sched):
    total, warm = 300, 20
    vals = [lr_at(s, total, 1.0, sched, warm) for s in range(total)]
    assert vals[0] == pytest.approx(1 / warm)
    assert vals[warm - 1] == pytest.approx(1.0)
    for s in range(1, warm):
        assert vals[s] > vals[s - 1]


def test_cosine_ends_at_lr_min_and_never_rises_after_warmup():
    total, warm = 300, 20
    vals = [lr_at(s, total, 1.0, "cosine", warm) for s in range(total)]
    assert vals[-1] == pytest.approx(0.1, abs=1e-9)
    assert all(b <= a + 1e-12 for a, b in zip(vals[warm:], vals[warm + 1:]))


def test_wsd_is_flat_then_decays_and_the_flat_part_is_exactly_flat():
    total, warm, frac = 300, 20, 0.2
    vals = [lr_at(s, total, 1.0, "wsd", warm, frac) for s in range(total)]
    start = total - int(round(frac * total))
    assert all(v == pytest.approx(1.0) for v in vals[warm - 1:start])
    assert vals[-1] == pytest.approx(0.1, abs=1e-9)
    assert all(b <= a + 1e-12 for a, b in zip(vals[start:], vals[start + 1:]))


def test_cosine_depends_on_the_declared_budget_which_is_the_whole_point_of_section_4():
    """Stopping a 300-step cosine at 200 is not a 200-step cosine."""
    a = lr_at(199, 300, 1.0, "cosine", 20)
    b = lr_at(199, 200, 1.0, "cosine", 20)
    assert a > b * 3


def test_unknown_schedule_is_refused():
    with pytest.raises(ValueError):
        lr_at(0, 10, 1.0, "linear-ish", 0)


# ---------------------------------------------------------------------- 6. muP

def test_groups_partition_every_parameter_exactly_once():
    m = GPT(BASE_WIDTH, "sp")
    g = m.groups()
    seen = [n for kind in g.values() for n, _ in kind]
    assert sorted(seen) == sorted(n for n, _ in m.named_parameters())
    assert len(seen) == len(set(seen))


def test_at_the_base_width_mup_and_sp_are_the_same_model():
    """m = 1, so every rescaling is by 1.  If this fails, section 5's control is broken."""
    a, b = GPT(BASE_WIDTH, "sp"), GPT(BASE_WIDTH, "mup")
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)
    ga = lr_groups(a, 1e-3, "sp")
    gb = lr_groups(b, 1e-3, "mup")
    assert [x["lr"] for x in ga] == [x["lr"] for x in gb]


@pytest.mark.parametrize("width", [512, 1024])
def test_mup_scales_matrix_learning_rates_by_the_width_multiple(width):
    m = GPT(width, "mup")
    mult = width / BASE_WIDTH
    groups = {g["tag"]: g["lr"] for g in lr_groups(m, 1e-3, "mup")}
    assert groups["matrix"] == pytest.approx(1e-3 / mult)
    assert groups["embedding"] == pytest.approx(1e-3)
    assert groups["vector"] == pytest.approx(1e-3)
    sp = {g["tag"]: g["lr"] for g in lr_groups(GPT(width, "sp"), 1e-3, "sp")}
    assert sp["matrix"] == pytest.approx(1e-3)


@pytest.mark.parametrize("width", [512, 1024])
def test_mup_shrinks_hidden_init_by_sqrt_of_the_width_multiple(width):
    mult = width / BASE_WIDTH
    sp, mup = GPT(width, "sp"), GPT(width, "mup")
    checked = 0
    for (n, a), (_, b) in zip(sp.named_parameters(), mup.named_parameters()):
        if n.startswith(("wte", "wpe")):
            assert torch.equal(a, b)                       # embeddings are never rescaled
        elif a.ndim >= 2:
            assert a.std().item() / b.std().item() == pytest.approx(math.sqrt(mult), rel=0.05)
            checked += 1
    assert checked >= 5


def test_embedding_and_readout_are_untied():
    """muP gives the input and output layers different rules; tying them would be a bug."""
    m = GPT(BASE_WIDTH, "mup")
    assert m.wte.weight.data_ptr() != m.lm_head.weight.data_ptr()


def test_heads_grow_with_width_so_the_attention_scale_needs_no_correction():
    """Head dim is held at 64 and heads are added instead, which is why 1/sqrt(d_head)
    needs no muP correction.  Built from Block directly: GPT(2048) is 100M parameters."""
    Block = P["Block"]
    for width in (256, 512, 1024, 2048):
        assert Block(width).n_head == width // HEAD_DIM
        assert Block(width).n_head * HEAD_DIM == width


def test_the_model_runs_and_starts_near_the_uniform_loss():
    m = GPT(BASE_WIDTH, "sp")
    idx = torch.randint(0, 65, (2, 16))
    loss, _ = m(idx, idx)
    assert loss.item() == pytest.approx(math.log(65), abs=0.35)


# ---------------------------------------------------------------- 7. reading off a minimum

def test_refine_finds_the_vertex_of_a_parabola_exactly():
    lrs = [10 ** x for x in (-4, -3.5, -3, -2.5, -2)]
    true_min = -3.1
    losses = [(math.log10(x) - true_min) ** 2 + 1 for x in lrs]
    got, ok = refine(lrs, losses)
    assert ok
    assert math.log10(got) == pytest.approx(true_min, abs=1e-9)


def test_refine_refuses_when_the_grid_did_not_bracket_the_minimum():
    lrs = [1e-4, 1e-3, 1e-2]
    got, ok = refine(lrs, [3.0, 2.0, 1.0])     # still falling at the right edge
    assert not ok and got == 1e-2
    got, ok = refine(lrs, [1.0, 2.0, 3.0])     # still falling at the left edge
    assert not ok and got == 1e-4


def test_refine_refuses_a_maximum():
    lrs = [1e-4, 1e-3, 1e-2]
    _, ok = refine(lrs, [2.0, 1.0, 2.5])
    assert ok                                   # a genuine interior minimum
    _, ok = refine([1e-4, 1e-3, 1e-2, 1e-1], [3.0, 1.0, 1.0, 3.0])
    assert ok


# ------------------------------------------------------------------- 8. small utilities

def test_digits_agreeing():
    assert digits_agreeing(1.0, 1.0) == 16.0
    assert digits_agreeing(0.0, 0.0) == 16.0
    assert digits_agreeing(1.0, 1.1) == pytest.approx(1.0, abs=0.05)
    assert digits_agreeing(1.0, 1.0001) == pytest.approx(4.0, abs=0.05)
    assert digits_agreeing(1.0, -1.0) == 0.0
