"""Gradient-correctness tests for the two reversible coupling schemes in reversible_llm.py.

These are the arithmetic claims, not measurements, and must hold on any machine: that the
euler (RevNet-style) custom backward and the midpoint (leapfrog) custom backward each produce
*exactly* the gradients an ordinary, memory-hungry autograd graph would have produced for the
same computation. A simulation that "looks reversible" but gets a gradient wrong would still
train something -- just not the model the forward pass describes. This is the torch version of
the check that was first done in pure NumPy (no torch available on the machine that wrote this)
in tools/verify_reversible_math.py; both must agree.
"""
import os
import tempfile

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_FILE = os.path.join(os.path.dirname(HERE), "reversible_llm.py")

# Importing the module executes its top-level cells (data download, model sizing, the full
# training runs) -- far too slow, and wrong, for a unit test. So instead exec only the
# pieces this file needs by reading the source and running everything up to the point the
# model classes are defined, into a throwaway namespace whose __file__ points at a scratch
# directory -- so its `results/` side effects land in a temp dir, not in the repo.
_SRC = open(REPO_FILE, encoding="utf-8").read()
_CUTOFF = _SRC.index("# %% [markdown]\n# ## 3. Sizing all three backbones")
_scratch = tempfile.mkdtemp(prefix="reversible_llm_test_")
_ns = {"__name__": "reversible_llm_defs", "__file__": os.path.join(_scratch, "reversible_llm.py")}
# section 1's data download needs the internet; tests don't need any real text, so stub a
# tiny corpus in before executing the module's top matter.
exec(compile(_SRC[:_CUTOFF].replace(
    'CORPUS = os.path.join(OUT_DIR, "tinyshakespeare.txt")',
    'CORPUS = os.path.join(OUT_DIR, "test_stub_corpus.txt")\n'
    'open(CORPUS, "w").write("to be or not to be that is the question " * 200)'
), "<reversible_llm-defs>", "exec"), _ns)

EulerBlockFunction = _ns["EulerBlockFunction"]
EulerCoupling = _ns["EulerCoupling"]
MidpointFunction = _ns["MidpointFunction"]
FullBlock = _ns["FullBlock"]
SubBlock = _ns["SubBlock"]

torch.manual_seed(0)


def _clone_module(m):
    import copy
    return copy.deepcopy(m)


# =========================================================================================
# euler / additive coupling
# =========================================================================================
def _build_euler_layers(n_layer, dim_half, n_head, block_size):
    return [EulerCoupling(dim_half, n_head, block_size, 0.0) for _ in range(n_layer)]


def test_euler_matches_plain_autograd():
    """The custom reversible backward must reproduce ordinary autograd exactly."""
    torch.manual_seed(1)
    B, T, dim_half, n_head, block_size, n_layer = 2, 5, 8, 2, 16, 3
    layers = _build_euler_layers(n_layer, dim_half, n_head, block_size)
    layers_ref = [_clone_module(l) for l in layers]

    x1 = torch.randn(B, T, dim_half, requires_grad=True)
    x2 = torch.randn(B, T, dim_half, requires_grad=True)
    x1_ref = x1.detach().clone().requires_grad_(True)
    x2_ref = x2.detach().clone().requires_grad_(True)

    # reversible path: chained custom Function, no intermediate activations retained
    a, b = x1, x2
    for layer in layers:
        a, b = layer(a, b)
    loss = (a.sum() + (b ** 2).sum())
    loss.backward()

    # reference path: identical math, plain autograd, everything retained
    a_ref, b_ref = x1_ref, x2_ref
    for layer in layers_ref:
        y1 = a_ref + layer.f(b_ref)
        y2 = b_ref + layer.g(y1)
        a_ref, b_ref = y1, y2
    loss_ref = (a_ref.sum() + (b_ref ** 2).sum())
    loss_ref.backward()

    assert torch.allclose(a, a_ref, atol=1e-5)
    assert torch.allclose(b, b_ref, atol=1e-5)
    assert torch.allclose(x1.grad, x1_ref.grad, atol=1e-4)
    assert torch.allclose(x2.grad, x2_ref.grad, atol=1e-4)

    for layer, layer_ref in zip(layers, layers_ref):
        for p, p_ref in zip(layer.parameters(), layer_ref.parameters()):
            assert p.grad is not None and p_ref.grad is not None
            assert torch.allclose(p.grad, p_ref.grad, atol=1e-4), \
                f"param grad mismatch, max diff {(p.grad - p_ref.grad).abs().max().item()}"


def test_euler_inversion_reconstructs_inputs():
    """x2 = y2 - G(y1); x1 = y1 - F(x2) must recover the exact forward inputs."""
    torch.manual_seed(2)
    dim_half, n_head, block_size = 8, 2, 16
    layer = EulerCoupling(dim_half, n_head, block_size, 0.0)
    x1 = torch.randn(2, 4, dim_half)
    x2 = torch.randn(2, 4, dim_half)
    with torch.no_grad():
        y1 = x1 + layer.f(x2)
        y2 = x2 + layer.g(y1)
        x2_rec = y2 - layer.g(y1)
        x1_rec = y1 - layer.f(x2_rec)
    assert torch.allclose(x2_rec, x2, atol=1e-6)
    assert torch.allclose(x1_rec, x1, atol=1e-6)


# =========================================================================================
# midpoint / leapfrog
# =========================================================================================
def test_midpoint_matches_plain_autograd():
    """The whole-stack custom Function must reproduce ordinary (fully materialized)
    autograd through the same leapfrog recurrence, including the P[-1]:=P[0] bootstrap
    term's extra contribution to grad(P0)."""
    torch.manual_seed(3)
    B, T, dim, n_head, block_size, n_layer, h = 2, 5, 8, 2, 16, 4, 0.2
    layers = [FullBlock(dim, n_head, block_size, 0.0) for _ in range(n_layer)]
    layers_ref = [_clone_module(l) for l in layers]

    P0 = torch.randn(B, T, dim, requires_grad=True)
    P0_ref = P0.detach().clone().requires_grad_(True)

    PL = MidpointFunction.apply(P0, h, layers)
    loss = (PL.sum() + (PL ** 2).sum() * 0.5)
    loss.backward()

    Pprev, Pcur = P0_ref, P0_ref
    for layer in layers_ref:
        Pnext = Pprev + 2 * h * layer(Pcur)
        Pprev, Pcur = Pcur, Pnext
    loss_ref = (Pcur.sum() + (Pcur ** 2).sum() * 0.5)
    loss_ref.backward()

    assert torch.allclose(PL, Pcur, atol=1e-5)
    assert P0.grad is not None and P0_ref.grad is not None
    assert torch.allclose(P0.grad, P0_ref.grad, atol=1e-4), \
        f"grad(P0) mismatch, max diff {(P0.grad - P0_ref.grad).abs().max().item()}"

    for layer, layer_ref in zip(layers, layers_ref):
        for p, p_ref in zip(layer.parameters(), layer_ref.parameters()):
            assert p.grad is not None and p_ref.grad is not None
            assert torch.allclose(p.grad, p_ref.grad, atol=1e-4), \
                f"param grad mismatch, max diff {(p.grad - p_ref.grad).abs().max().item()}"


def test_midpoint_inversion_reconstructs_states():
    """P[l-1] = P[l+1] - 2h f_l(P[l]) must recover every intermediate state, which is the
    property that lets the real run avoid storing them at all."""
    torch.manual_seed(4)
    dim, n_head, block_size, n_layer, h = 8, 2, 16, 5, 0.2
    layers = [FullBlock(dim, n_head, block_size, 0.0) for _ in range(n_layer)]
    P0 = torch.randn(2, 4, dim)

    states = [P0]
    with torch.no_grad():
        Pprev, Pcur = P0, P0
        for layer in layers:
            Pnext = Pprev + 2 * h * layer(Pcur)
            states.append(Pnext)
            Pprev, Pcur = Pcur, Pnext

    # walk backward re-deriving each earlier state and check it against the forward record
    with torch.no_grad():
        for l in range(n_layer, 0, -1):
            f_out = layers[l - 1](states[l - 1])
            recon = states[l] - 2 * h * f_out
            expected = states[l - 2] if l - 2 >= 0 else states[0]
            assert torch.allclose(recon, expected, atol=1e-6), f"failed to invert step {l}"


def test_midpoint_gradient_needs_bootstrap_correction():
    """Regression guard for the easiest bug in this scheme: dropping the extra dL/dP1 term
    that P0 picks up from also playing P[-1] in the very first step. With it omitted the
    computed grad(P0) is wrong whenever the stack has more than one layer."""
    torch.manual_seed(5)
    dim, n_head, block_size, n_layer, h = 8, 2, 16, 3, 0.3
    layers = [FullBlock(dim, n_head, block_size, 0.0) for _ in range(n_layer)]
    P0 = torch.randn(2, 3, dim, requires_grad=True)

    PL = MidpointFunction.apply(P0, h, layers)
    loss = PL.sum()
    loss.backward()
    grad_with_correction = P0.grad.clone()

    # recompute by hand WITHOUT the bootstrap correction, to show it actually matters
    P0b = P0.detach().clone().requires_grad_(True)
    with torch.no_grad():
        Pprev, Pcur = P0b, P0b
        states = [P0b]
        for layer in layers:
            Pnext = Pprev + 2 * h * layer(Pcur)
            states.append(Pnext)
            Pprev, Pcur = Pcur, Pnext
    grad_ext = torch.ones_like(states[-1])
    g_next = grad_ext
    g_next2 = torch.zeros_like(grad_ext)
    cur, nxt = states[-2], states[-1]
    for l in range(n_layer - 1, -1, -1):
        cur_ = cur.detach().requires_grad_(True)
        f_out = layers[l](cur_)
        torch.autograd.backward(f_out, 2 * h * g_next, retain_graph=False)
        g_l = cur_.grad + g_next2
        with torch.no_grad():
            prev = nxt - 2 * h * f_out.detach()
        nxt, cur = cur, prev
        g_next2, g_next = g_next, g_l
    grad_without_correction = g_next  # deliberately omits + g1_alias

    assert not torch.allclose(grad_without_correction, grad_with_correction, atol=1e-4), \
        "the bootstrap correction should change grad(P0) whenever n_layer > 1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
