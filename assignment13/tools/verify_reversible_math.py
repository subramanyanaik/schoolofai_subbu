"""
Pure-NumPy verification of the two reversible coupling schemes, written and run *before* the
torch version in ../reversible_llm.py. The machine this repo was written on blocks torch's
native DLLs under a Windows Application Control policy (confirmed: CPU-only torch fails to
import there with WinError 4551, independent of install path), so this script hand-derives
forward + manual backward for both schemes in plain NumPy and checks them against
central-difference finite gradients, with no torch dependency at all:

  1. "euler"    -- additive coupling / RevNet-style two-stream reversible block
                   y1 = x1 + F(x2) ; y2 = x2 + G(y1)
  2. "midpoint" -- single-stream leapfrog recurrence
                   P[l+1] = P[l-1] + 2h*f_l(P[l]),  bootstrapped with P[-1] := P[0]

Both must match finite differences to ~1e-6 or the torch autograd.Function implementations in
reversible_llm.py are wrong before they are ever run on a GPU. The torch versions are checked
against ordinary (non-reversible) autograd directly in tests/test_reversible_math.py, which
runs in CI on every push; this script is the from-scratch derivation that test is built from,
kept for anyone who wants to see the math with nothing but NumPy between them and it.

    python tools/verify_reversible_math.py
"""
import numpy as np

rng = np.random.default_rng(0)
TOL = 1e-6


class TinyMLP:
    """f(x) = tanh(x @ W + b), with its own manual vjp. Stands in for "attention" or "mlp"
    in the real model -- any smooth function works for this check."""

    def __init__(self, d, seed):
        r = np.random.default_rng(seed)
        self.W = r.normal(scale=0.5, size=(d, d))
        self.b = r.normal(scale=0.1, size=(d,))

    def forward(self, x):
        z = x @ self.W + self.b
        return np.tanh(z), z

    def backward(self, x, z, grad_out):
        dz = grad_out * (1 - np.tanh(z) ** 2)
        dx = dz @ self.W.T
        dW = x.T @ dz
        db = dz.sum(axis=0)
        return dx, dW, db


def numeric_grad(fn, x, eps=1e-5):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f1 = fn()
        x[idx] = old - eps
        f2 = fn()
        x[idx] = old
        g[idx] = (f1 - f2) / (2 * eps)
    return g


def check(name, manual, numeric, tol=TOL):
    err = float(np.max(np.abs(manual - numeric)))
    status = "OK" if err < tol else "FAIL"
    print(f"  [{status}] {name}: max err {err:.3e}")
    assert err < tol, f"{name}: manual vs numeric gradient mismatch ({err:.3e} >= {tol:.3e})"


def verify_euler(B=3, D=4, L=3):
    print("=== euler (additive coupling) ===")
    Fs = [TinyMLP(D, seed=100 + i) for i in range(L)]
    Gs = [TinyMLP(D, seed=200 + i) for i in range(L)]
    x1_0 = rng.normal(size=(B, D))
    x2_0 = rng.normal(size=(B, D))
    loss_w = rng.normal(size=(B, D))

    def forward(x1, x2, save=False):
        stash = []
        for l in range(L):
            f_out, f_z = Fs[l].forward(x2)
            y1 = x1 + f_out
            g_out, g_z = Gs[l].forward(y1)
            y2 = x2 + g_out
            if save:
                stash.append((x2, f_z, y1, g_z))
            x1, x2 = y1, y2
        return x1, x2, stash

    def loss_fn(x1, x2):
        y1, y2, _ = forward(x1, x2)
        return float(np.sum(loss_w * y1) + np.sum(y2 ** 2) * 0.5)

    y1_final, y2_final, stash = forward(x1_0, x2_0, save=True)
    grad_y1, grad_y2 = loss_w.copy(), y2_final.copy()
    param_grads = {}

    for l in reversed(range(L)):
        x2_saved, f_z, y1, g_z = stash[l]
        dG_in, dGW, dGb = Gs[l].backward(y1, g_z, grad_y2)
        grad_y1_total = grad_y1 + dG_in
        # x2_saved *is* x2 at this layer (the inverse x2 = y2 - G(y1) reproduces it exactly,
        # so there is nothing to recompute -- the stash already holds the value backward needs)
        dF_in, dFW, dFb = Fs[l].backward(x2_saved, f_z, grad_y1_total)
        grad_x2_total = grad_y2 + dF_in
        param_grads[("G", l)] = dGW
        param_grads[("F", l)] = dFW
        grad_y1, grad_y2 = grad_y1_total, grad_x2_total

    grad_x1_manual, grad_x2_manual = grad_y1, grad_y2
    check("grad_x1", grad_x1_manual, numeric_grad(lambda: loss_fn(x1_0, x2_0), x1_0))
    check("grad_x2", grad_x2_manual, numeric_grad(lambda: loss_fn(x1_0, x2_0), x2_0))
    for l in range(L):
        check(f"layer {l} F.W", param_grads[("F", l)],
              numeric_grad(lambda: loss_fn(x1_0, x2_0), Fs[l].W))
        check(f"layer {l} G.W", param_grads[("G", l)],
              numeric_grad(lambda: loss_fn(x1_0, x2_0), Gs[l].W))


def verify_midpoint(B=3, D=4, L=3, h=0.3):
    print("\n=== midpoint (leapfrog) ===")
    fs = [TinyMLP(D, seed=300 + i) for i in range(L)]
    loss_w = rng.normal(size=(B, D))
    P0_init = rng.normal(size=(B, D))

    def forward(P0):
        Pprev, Pcur = P0, P0  # bootstrap P[-1] := P[0]
        states = [Pcur]
        for l in range(L):
            f_out, _ = fs[l].forward(Pcur)
            Pnext = Pprev + 2 * h * f_out
            states.append(Pnext)
            Pprev, Pcur = Pcur, Pnext
        return states

    def loss_fn(P0):
        PL = forward(P0)[-1]
        return float(np.sum(loss_w * PL) + np.sum(PL ** 2) * 0.5)

    states = forward(P0_init)
    grad_ext = loss_w + states[-1]
    g_next, g_next2 = grad_ext, np.zeros_like(grad_ext)
    param_grads, g1_alias = {}, None

    for l in range(L - 1, -1, -1):
        if l == 0:
            g1_alias = g_next  # dL/dP1, needed for the P[-1]:=P[0] bootstrap correction
        Pl = states[l]
        f_out, z = fs[l].forward(Pl)
        dPl, dW, db = fs[l].backward(Pl, z, 2 * h * g_next)
        param_grads[l] = dW
        g_l = dPl + g_next2
        g_next2, g_next = g_next, g_l

    grad_P0_manual = g_next + g1_alias
    check("grad_P0", grad_P0_manual, numeric_grad(lambda: loss_fn(P0_init), P0_init))
    for l in range(L):
        check(f"layer {l} f.W", param_grads[l], numeric_grad(lambda: loss_fn(P0_init), fs[l].W))

    # the property that lets the real implementation avoid storing any of these: every
    # intermediate state must be exactly recoverable from just the pair after it.
    ok = True
    for l in range(L, 0, -1):
        f_out, _ = fs[l - 1].forward(states[l - 1])
        recon_prev = states[l] - 2 * h * f_out
        expected = states[l - 2] if l - 2 >= 0 else states[0]
        ok &= bool(np.allclose(recon_prev, expected, atol=1e-10))
    print(f"  [{'OK' if ok else 'FAIL'}] inverse reconstructs every intermediate state")
    assert ok


if __name__ == "__main__":
    verify_euler()
    verify_midpoint()
    print("\nall checks passed.")
