"""The Kronecker V2 codec: bytes -> fixed vector, and back again exactly.

V1 (arXiv:2605.29459 Eq. 1) is the special case G = I_256, Phi = I_L.  Both of
its codebooks are the identity, and that single fact is the cause of D = 8192,
the 32-character cap, and the sparsity waste.  V2 replaces both codebooks:

    M_pos(v)[c, p] = G[b_p, c] / sqrt(L)     for p < L, else 0      (d_c x L_max)
    K(v)           = vec(M_pos(v) @ Phi)                            (D = d_c*L_max)

Theorem (exact inversion).  Phi is orthogonal, so M_pos = M_freq @ Phi.T holds
exactly for every L <= L_max -- one matmul, condition number 1, no pseudo-inverse.
Length falls out of the decode because the zero vector is reserved as "absent":
L is the first column whose norm drops below threshold.
"""

from __future__ import annotations

import numpy as np

from .codebooks import byte_codebook, coherence_stats, position_basis

ABSENT_THRESHOLD = 1e-3


class KroneckerCodec:
    """Deterministic byte codec.  No learned parameters, float64 throughout.

    ``variant="v1"`` pins the published configuration (d_c=256 one-hot byte
    codes, identity position basis) so V1 and V2 can be measured side by side
    at matched D.
    """

    def __init__(
        self,
        d_c: int = 64,
        L_max: int = 64,
        seed: int = 0,
        byte_method: str = "hadamard_bits",
        pos_kind: str = "dft",
        variant: str = "v2",
        overflow: str = "truncate",
    ) -> None:
        if variant not in ("v1", "v2"):
            raise ValueError(f"variant must be 'v1' or 'v2', got {variant!r}")
        if overflow not in ("truncate", "alias"):
            raise ValueError(f"overflow must be 'truncate' or 'alias', got {overflow!r}")
        if variant == "v1":
            d_c, byte_method, pos_kind, overflow = 256, "onehot", "identity", "truncate"

        self.variant = variant
        self.d_c = int(d_c)
        self.L_max = int(L_max)
        self.seed = int(seed)
        self.byte_method = byte_method
        self.pos_kind = pos_kind
        self.overflow = overflow

        self.G = byte_codebook(self.d_c, seed=self.seed, method=byte_method)
        self.Phi = position_basis(self.L_max, kind=pos_kind)
        self.D = self.d_c * self.L_max

    # ---------------------------------------------------------------- helpers

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"KroneckerCodec(variant={self.variant}, d_c={self.d_c}, "
            f"L_max={self.L_max}, D={self.D}, byte={self.byte_method}, "
            f"pos={self.pos_kind}, overflow={self.overflow})"
        )

    def coherence(self) -> dict:
        return coherence_stats(self.G)

    def effective_lengths(self, seqs) -> np.ndarray:
        """Number of bytes each sequence actually contributes."""
        lens = np.array([len(s) for s in seqs], dtype=np.int64)
        if self.overflow == "truncate":
            return np.minimum(lens, self.L_max)
        return lens

    # ---------------------------------------------------------------- encode

    def encode_positions(self, seqs) -> np.ndarray:
        """(N, d_c, L_max) float64 position-domain codec matrices.

        Trap: the readable per-sequence loop takes ~105 s on the GPT-2 vocab.
        This builds one flat scatter instead.  ``np.add.at`` is only used when
        some sequence actually overflows in alias mode, because aliased slots
        must accumulate; plain fancy-index assignment is far faster otherwise.
        """
        seqs = list(seqs)
        N = len(seqs)
        L_max, d_c = self.L_max, self.d_c
        out = np.zeros((N, L_max, d_c), dtype=np.float64)
        if N == 0:
            return out.transpose(0, 2, 1)

        if self.overflow == "truncate":
            kept = [bytes(s)[:L_max] for s in seqs]
        else:
            kept = [bytes(s) for s in seqs]

        lens = np.array([len(s) for s in kept], dtype=np.int64)
        total = int(lens.sum())
        if total:
            flat = np.frombuffer(b"".join(kept), dtype=np.uint8)
            row = np.repeat(np.arange(N, dtype=np.int64), lens)
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            pos = (np.arange(total, dtype=np.int64) - np.repeat(starts, lens)) % L_max
            codes = self.G[flat]
            if lens.max() > L_max:  # alias mode with real overflow
                np.add.at(out, (row, pos), codes)
            else:
                out[row, pos] = codes

        scale = np.zeros(N, dtype=np.float64)
        nz = lens > 0
        scale[nz] = 1.0 / np.sqrt(lens[nz])
        out *= scale[:, None, None]
        return np.ascontiguousarray(out.transpose(0, 2, 1))

    def encode(self, seqs) -> np.ndarray:
        """(N, D) float64 codec vectors, K = vec(M_pos @ Phi)."""
        M = self.encode_positions(seqs)
        K = M @ self.Phi
        return K.reshape(K.shape[0], self.D)

    # ---------------------------------------------------------------- decode

    def decode_positions(self, K: np.ndarray) -> np.ndarray:
        """Invert the position basis.  Exact: Phi is orthogonal."""
        K = np.asarray(K, dtype=np.float64)
        M_freq = K.reshape(-1, self.d_c, self.L_max)
        return M_freq @ self.Phi.T

    def decode(self, K: np.ndarray, method: str | None = None) -> list[bytes]:
        """Recover the byte strings.  ``method`` in {auto, sign, nn}."""
        if method is None or method == "auto":
            method = "sign" if self.byte_method == "hadamard_bits" else "nn"

        M_pos = self.decode_positions(K)  # (N, d_c, L_max)
        N = M_pos.shape[0]
        norms = np.linalg.norm(M_pos, axis=1)  # (N, L_max)
        present = norms > ABSENT_THRESHOLD

        # Length is the first absent column.  Positions are contiguous by
        # construction, so this is exact rather than a heuristic.
        lengths = np.where(present.all(axis=1), self.L_max, np.argmin(present, axis=1))

        # Rescale so every present column is a unit-norm byte code again.
        scaled = M_pos * np.sqrt(np.maximum(lengths, 1))[:, None, None]

        if method == "sign":
            bits = (scaled[:, :8, :] < 0).astype(np.int64)
            weights = (1 << np.arange(8, dtype=np.int64))[None, :, None]
            vals = (bits * weights).sum(axis=1)  # (N, L_max)
        elif method == "nn":
            if self.byte_method == "onehot":  # G = I, so the search is an argmax
                vals = np.argmax(scaled, axis=1)
            else:
                vals = np.empty((N, self.L_max), dtype=np.int64)
                step = max(1, 2_000_000 // (256 * self.L_max))
                for lo in range(0, N, step):
                    blk = scaled[lo : lo + step]
                    vals[lo : lo + step] = np.argmax(
                        np.tensordot(self.G, blk, axes=([1], [1])), axis=0
                    )
        else:
            raise ValueError(f"unknown decode method {method!r}")

        vals = np.clip(vals, 0, 255).astype(np.uint8)
        return [vals[i, : lengths[i]].tobytes() for i in range(N)]

    # ------------------------------------------------------------- roundtrip

    def roundtrip(self, seqs, method: str | None = None) -> list[bytes]:
        return self.decode(self.encode(seqs), method=method)

    def roundtrip_stats(self, seqs, chunk: int = 4096, method: str | None = None) -> dict:
        """Chunked round-trip over a large corpus of byte strings.

        Chunking is not optional at vocabulary scale: (50257, 8192) in float64
        is 3.3 GB, and E1 needs the answer, not the array.
        """
        seqs = [bytes(s) for s in seqs]
        exact = 0
        exact_within = 0
        within = 0
        byte_hits = 0
        byte_total = 0
        failures: list[tuple[bytes, bytes]] = []
        for start in range(0, len(seqs), chunk):
            block = seqs[start : start + chunk]
            got = self.roundtrip(block, method=method)
            for src, dst in zip(block, got):
                fits = len(src) <= self.L_max
                within += int(fits)
                ok = src == dst
                exact += int(ok)
                exact_within += int(ok and fits)
                n = min(len(src), len(dst))
                byte_hits += sum(1 for a, b in zip(src[:n], dst[:n]) if a == b)
                byte_total += max(len(src), 1)
                if not ok and len(failures) < 10:
                    failures.append((src, dst))
        n_total = len(seqs)
        return {
            "n": n_total,
            "exact": exact,
            "exact_frac": exact / n_total if n_total else 1.0,
            "within_L_max": within,
            "exact_within_L_max": exact_within,
            "exact_frac_within_L_max": exact_within / within if within else 1.0,
            "byte_accuracy": byte_hits / byte_total if byte_total else 1.0,
            "failures": [(s.hex(), d.hex()) for s, d in failures],
        }
