"""Deterministic, zero-parameter codebooks for the Kronecker V2 codec.

Everything in this module is built in float64 and contains no learned state.

Trap 1 (see README III.4): these are *constructed* in float64, never built in
float32 and cast.  Casting caps end-to-end exactness at ~5e-8 and the fp64
acceptance gate then fails for a reason that looks like a maths bug.  The single
downcast to the training dtype happens in ``KASHead.__init__``.
"""

from __future__ import annotations

import numpy as np

BIT_COLUMNS = (1, 2, 4, 8, 16, 32, 64, 128)
"""The 8 Walsh bit-columns of the Sylvester-Hadamard matrix.

Column ``2**k`` of H satisfies ``H[i, 2**k] = (-1)**bit_k(i)``, so coordinate k
of row i has the sign of bit k of the byte value i.  Forcing these 8 columns
into every codebook makes all 256 rows provably distinct (the byte is recovered
from the signs) and turns decoding into an O(8) sign read.
"""


def sylvester_hadamard(n: int = 256) -> np.ndarray:
    """H[i, j] = (-1)**popcount(i & j), the Sylvester-Hadamard matrix.

    Satisfies H @ H.T = n * I.  ``n`` must be a power of two.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of two, got {n}")
    idx = np.arange(n, dtype=np.int64)
    both = idx[:, None] & idx[None, :]
    # popcount via the standard SWAR trick, vectorised over the whole matrix.
    popcount = np.zeros_like(both)
    x = both.copy()
    while np.any(x):
        popcount += x & 1
        x >>= 1
    return np.where(popcount & 1, -1.0, 1.0).astype(np.float64)


def byte_codebook(
    d_c: int,
    seed: int = 0,
    method: str = "hadamard_bits",
) -> np.ndarray:
    """Return a (256, d_c) float64 codebook with unit-norm rows.

    method:
      ``hadamard_bits``   d_c columns of the 256x256 Sylvester-Hadamard matrix,
                          forced to contain the 8 Walsh bit-columns first, the
                          remainder drawn without replacement using ``seed``.
                          Requires d_c >= 8.  This is the V2 default.
      ``hadamard_random`` d_c Hadamard columns with no forced bit-columns.
                          Rows are *not* guaranteed distinct; used as a foil.
      ``rademacher``      iid +-1 entries.
      ``onehot``          the identity, i.e. exactly V1's byte code.  d_c = 256.
    """
    if method == "onehot":
        if d_c != 256:
            raise ValueError("onehot byte code requires d_c == 256 (this is V1)")
        return np.eye(256, dtype=np.float64)

    if method == "rademacher":
        rng = np.random.default_rng(seed)
        G = rng.choice([-1.0, 1.0], size=(256, d_c)).astype(np.float64)
        return G / np.sqrt(d_c)

    H = sylvester_hadamard(256)

    if method == "hadamard_random":
        rng = np.random.default_rng(seed)
        cols = rng.choice(np.arange(1, 256), size=d_c, replace=False)
        return H[:, cols] / np.sqrt(d_c)

    if method != "hadamard_bits":
        raise ValueError(f"unknown byte codebook method {method!r}")

    if d_c < len(BIT_COLUMNS):
        raise ValueError(f"hadamard_bits needs d_c >= {len(BIT_COLUMNS)}, got {d_c}")
    if d_c > 255:
        raise ValueError("hadamard_bits draws from the 255 non-constant columns")

    # Column 0 of H is all-ones.  It carries no information about the byte and
    # adds a constant +1 to every pairwise inner product, i.e. it raises
    # coherence for free, so it is excluded from the candidate pool.
    forced = np.array(BIT_COLUMNS, dtype=np.int64)
    pool = np.setdiff1d(np.arange(1, 256, dtype=np.int64), forced)
    rng = np.random.default_rng(seed)
    extra = rng.permutation(pool)[: d_c - len(forced)]
    cols = np.concatenate([forced, np.sort(extra)])
    return H[:, cols] / np.sqrt(d_c)


def position_basis(L_max: int, kind: str = "dft") -> np.ndarray:
    """Return an orthogonal (L_max, L_max) float64 position basis Phi.

    Convention: ``Phi[p, k]`` is basis function k evaluated at position p, so a
    position-domain matrix is transformed by ``M_pos @ Phi`` and inverted by
    ``M_freq @ Phi.T``.  Phi.T @ Phi = Phi @ Phi.T = I either way.

    ``dft``      real orthonormal Fourier basis with period P = L_max.  The
                 period is load-bearing: a longer period would squeeze
                 positions 0..L-1 into a small arc of the circle and the system
                 becomes ill-conditioned.  Column 0 is the constant 1/sqrt(N);
                 for k = 1..N/2-1 a cos/sin pair at sqrt(2/N); for even N a
                 final Nyquist column (-1)**p / sqrt(N).
    ``identity`` exactly V1: positions are their own basis.
    """
    N = int(L_max)
    if N <= 0:
        raise ValueError("L_max must be positive")

    if kind == "identity":
        return np.eye(N, dtype=np.float64)
    if kind != "dft":
        raise ValueError(f"unknown position basis kind {kind!r}")

    p = np.arange(N, dtype=np.float64)
    cols = [np.full(N, 1.0 / np.sqrt(N))]
    k_top = (N - 1) // 2  # last k that contributes a full cos/sin pair
    for k in range(1, k_top + 1):
        ang = 2.0 * np.pi * k * p / N
        cols.append(np.sqrt(2.0 / N) * np.cos(ang))
        cols.append(np.sqrt(2.0 / N) * np.sin(ang))
    if N % 2 == 0 and N > 1:
        cols.append(np.cos(np.pi * p) / np.sqrt(N))
    Phi = np.stack(cols, axis=1)
    assert Phi.shape == (N, N), (Phi.shape, N)
    return np.ascontiguousarray(Phi)


def welch_bound(n: int, d: int) -> float:
    """Welch lower bound on the max coherence of n unit vectors in R^d."""
    if n <= d:
        return 0.0
    return float(np.sqrt((n - d) / (d * (n - 1))))


def coherence_stats(G: np.ndarray) -> dict:
    """Coherence of a unit-norm-row codebook.

    E5 measures this to be the quantity that orders language-model quality --
    not D, and not d_c directly.  It is closed form, so a codec configuration
    can be chosen analytically instead of by sweeping.
    """
    n, d = G.shape
    norms = np.linalg.norm(G, axis=1)
    Gn = G / norms[:, None]
    gram = np.abs(Gn @ Gn.T)
    np.fill_diagonal(gram, 0.0)
    # Rows are distinct iff no pair has |cos| == 1 with the same sign, but the
    # honest check is on the raw rows.
    uniq = np.unique(G, axis=0)
    return {
        "n": int(n),
        "d": int(d),
        "max_coherence": float(gram.max()),
        "mean_coherence": float(gram.sum() / (n * (n - 1))),
        "welch_bound": welch_bound(n, d),
        "rows_distinct": bool(uniq.shape[0] == n),
        "min_row_norm": float(norms.min()),
        "max_row_norm": float(norms.max()),
    }
