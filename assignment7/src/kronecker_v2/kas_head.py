"""KAS -- Kronecker Additive Softmax.  The output head with no vocabulary.

The claim in one line.  Because the codec is *linear in byte-position
occupancy* -- and no other surface-form encoder is -- the full-vocabulary logit
vector is a sum of scalar lookups instead of |V| d-dimensional dot products:

    logit(v) = <S_pos, M_pos(v)>_F = (1/sqrt(L_v)) * sum_{p<L_v} A[b_p, p],
    A        = G @ S_pos,   a 256 x L_max table, one small precompute per batch.

Equivalently, and this is how the code actually runs: the dense output weight
matrix is replaced by the vocabulary's byte-occupancy matrix -- fixed, derived,
zero trainable parameters, ~0.16% dense.

This is an *exact reformulation*, not an approximation.  Logits, the partition
function and gradients are bit-identical to the dense head, unlike hierarchical,
adaptive or sampled softmax which define a different distribution.  That is the
most likely misreading of the work, so it is enforced by a test (E2).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMPLS = ("bag", "sparse", "loop", "dense")


class Occupancy(nn.Module):
    """The byte-occupancy matrix of a vocabulary, as buffers only.

    Trap 5: this is a module purely so its buffers move with ``.to(device)``.
    It must have **no parameters**.  Assigning a whole ``KASHead`` somewhere
    just to borrow its buffers silently registers that head's unused ``W_out``
    as a parameter of the host module; it never gets a gradient so training is
    unaffected and *only the reported parameter count is wrong* -- which is the
    single number this project is about.  ``test_no_phantom_parameters``
    guards it.
    """

    def __init__(self, packed, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.V = packed.V
        self.L_max = packed.L_max
        self.n_cells = 256 * packed.L_max
        self.nnz = packed.total_bytes

        cols = packed.flat_bytes.astype(np.int64) * packed.L_max + packed.pos_mod
        vals = np.repeat(packed.inv_sqrt_len, packed.lengths)

        self.register_buffer("cols", torch.from_numpy(cols), persistent=False)
        self.register_buffer(
            "vals", torch.from_numpy(vals).to(dtype), persistent=False
        )
        self.register_buffer(
            "offsets", torch.from_numpy(packed.offsets.copy()), persistent=False
        )
        self.register_buffer(
            "lengths", torch.from_numpy(packed.lengths.copy()), persistent=False
        )
        self.register_buffer(
            "inv_sqrt_len",
            torch.from_numpy(packed.inv_sqrt_len.copy()).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "hi", torch.from_numpy(packed.hi.copy()), persistent=False
        )
        # Bytes at position p, for the contiguous suffix of tokens that reach p.
        self._pos_bytes: list[torch.Tensor] = []
        for p in range(packed.L_max):
            self._pos_bytes.append(torch.from_numpy(packed.bytes_at(p)))
        for i, t in enumerate(self._pos_bytes):
            self.register_buffer(f"pos_bytes_{i}", t, persistent=False)

    def pos_bytes(self, p: int) -> torch.Tensor:
        return getattr(self, f"pos_bytes_{p}")

    @property
    def density(self) -> float:
        return self.nnz / (self.V * self.n_cells)

    def matmul(self, dense: torch.Tensor) -> torch.Tensor:
        """(256*L_max, k) -> (V, k).  The fast path.

        Trap 2: ``torch.sparse.mm`` cannot mix an fp32 sparse operand with a
        bf16 dense one -- under ``autocast(bfloat16)`` cuSPARSE raises
        ``cusparseSpMM_bufferSize ... not supported``.  ``F.embedding_bag`` is
        a fused kernel that works fine under autocast.

        Trap 3: never form ``A[:, all_vocab_bytes, all_pos]``.  That gather
        materialises N x B_V (~2e9 floats at N=8192).
        """
        w = dense.contiguous()
        return F.embedding_bag(
            self.cols,
            w,
            self.offsets,
            mode="sum",
            per_sample_weights=self.vals.to(w.dtype),
        )

    def to_sparse_csr(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        idx = torch.stack([
            torch.repeat_interleave(
                torch.arange(self.V, device=self.cols.device), self.lengths
            ),
            self.cols,
        ])
        vals = self.vals if dtype is None else self.vals.to(dtype)
        coo = torch.sparse_coo_tensor(idx, vals, (self.V, self.n_cells)).coalesce()
        return coo.to_sparse_csr()

    def to_dense_matrix(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        out = torch.zeros(
            self.V, self.n_cells, dtype=dtype or self.vals.dtype, device=self.cols.device
        )
        rows = torch.repeat_interleave(
            torch.arange(self.V, device=self.cols.device), self.lengths
        )
        out.index_put_((rows, self.cols), self.vals.to(out.dtype), accumulate=True)
        return out


def dense_codec_table(packed, codec, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """(V, d_c*L_max) position-domain codec table -- the naive output matrix.

    This is the thing KAS must reproduce bit-for-bit.  It is the honest
    reference: a plain dense matrix of the codec vectors, used with a plain
    dense matmul.
    """
    M = codec.encode_positions(packed.token_bytes)  # (V, d_c, L_max)
    return torch.from_numpy(M.reshape(packed.V, -1)).to(dtype)


class KASHead(nn.Module):
    """Vocabulary-free output head.

    Parameters: ``W_out`` (d_model x D) -- or nothing at all when tied -- plus
    ``beta``, one scalar per token *length* (not per token), plus an optional
    unigram bias of one scalar per token.  Nothing here scales with |V| except
    the optional unigram bias, and that is exactly the point of E4/A4.
    """

    def __init__(
        self,
        d_model: int,
        codec,
        packed,
        tied_weight: nn.Parameter | None = None,
        dtype: torch.dtype = torch.float32,
        impl: str = "bag",
        unigram_bias: bool = False,
        occ: Occupancy | None = None,
    ) -> None:
        super().__init__()
        if impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
        self.d_model = int(d_model)
        self.d_c = codec.d_c
        self.L_max = codec.L_max
        self.D = codec.D
        self.V = packed.V
        self.impl = impl

        # The single float64 -> training dtype boundary (trap 1).
        self.register_buffer("G", torch.from_numpy(codec.G.copy()).to(dtype))

        if tied_weight is None:
            w = torch.randn(self.d_model, self.D, dtype=dtype) * (self.D ** -0.5)
            self.W_out = nn.Parameter(w)
            self.tied = None
        else:
            # (D, d_model), owned by the input embedding.  Registering it here
            # too is safe: named_parameters() de-duplicates by identity.
            self.W_out = None
            self.tied = tied_weight

        # Sized by L_max, never by the observed max token length: beta must be
        # a function of the *codec configuration* only.  Sizing it from the
        # vocabulary would make the head's parameter count creep with |V|
        # (params_audit caught exactly that), which is the one thing C1 forbids.
        self.beta = nn.Parameter(torch.ones(self.L_max + 1, dtype=dtype))
        self.unigram = (
            nn.Parameter(torch.zeros(self.V, dtype=dtype)) if unigram_bias else None
        )

        self.occ = occ if occ is not None else Occupancy(packed, dtype=dtype)
        self._dense_table: torch.Tensor | None = None
        self._csr: torch.Tensor | None = None

    # ------------------------------------------------------------------ util

    @property
    def is_tied(self) -> bool:
        return self.tied is not None

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """h (N, d_model) -> S (N, D)."""
        if self.tied is None:
            return h @ self.W_out
        return h @ self.tied.t()

    def head_parameters(self) -> int:
        """Parameters this head owns that a dense head would not amortise."""
        n = 0
        if self.W_out is not None:
            n += self.W_out.numel()
        n += self.beta.numel()
        if self.unigram is not None:
            n += self.unigram.numel()
        return n

    # -------------------------------------------------------------- the head

    def logits_from_A(self, A_T: torch.Tensor, impl: str) -> torch.Tensor:
        """A_T (256*L_max, N) -> logits (N, V)."""
        if impl == "bag":
            return self.occ.matmul(A_T).t()
        if impl == "sparse":
            if self._csr is None or self._csr.dtype != A_T.dtype:
                self._csr = self.occ.to_sparse_csr(A_T.dtype)
            return torch.sparse.mm(self._csr, A_T).t()
        raise ValueError(impl)

    def forward(self, h: torch.Tensor, impl: str | None = None) -> torch.Tensor:
        impl = impl or self.impl
        flat = h.reshape(-1, self.d_model)
        N = flat.shape[0]
        S = self.project(flat).view(N, self.d_c, self.L_max)

        if impl == "dense":
            if self._dense_table is None:
                raise RuntimeError("call attach_dense_table() before impl='dense'")
            table = self._dense_table.to(S.dtype)
            logits = S.reshape(N, -1) @ table.t()
        elif impl == "loop":
            logits = self._loop_logits(S)
        else:
            # Build A transposed so the (256*L_max, N) view is free -- no copy.
            A_T = torch.einsum("vc,ncp->vpn", self.G.to(S.dtype), S)
            A_T = A_T.reshape(256 * self.L_max, N)
            logits = self.logits_from_A(A_T, impl)

        scale = self.beta[self.occ.lengths].to(logits.dtype)
        if torch.is_grad_enabled():
            logits = logits * scale
            if self.unigram is not None:
                logits = logits + self.unigram.to(logits.dtype)
        else:
            # Numerically identical, but avoids two full (N, V) temporaries.
            # At |V| = 1M and N = 256 that is 1 GB apiece -- the difference
            # between E6 fitting on an 4 GB card and not.
            logits = logits.mul_(scale)
            if self.unigram is not None:
                logits = logits.add_(self.unigram.to(logits.dtype))
        return logits.view(*h.shape[:-1], self.V)

    def _loop_logits(self, S: torch.Tensor) -> torch.Tensor:
        """The obvious implementation, kept as evidence the fast path is honest.

        Trap 4: on GPU this launches L_max tiny kernels and is latency-bound --
        it measured ~20x *slower* than the dense head.  Recognising the same
        operation as a sparse matmul is what made KAS practical.
        """
        N = S.shape[0]
        A = torch.einsum("vc,ncp->nvp", self.G.to(S.dtype), S)  # (N, 256, L_max)
        out = torch.zeros(N, self.V, dtype=S.dtype, device=S.device)
        for p in range(self.L_max):
            lo = int(self.occ.hi[p])
            if lo >= self.V:
                continue
            b = self.occ.pos_bytes(p)
            out[:, lo:] = out[:, lo:] + A[:, b, p]
        return out * self.occ.inv_sqrt_len.to(out.dtype)

    # ------------------------------------------------------------ references

    def attach_dense_table(self, table: torch.Tensor) -> None:
        self._dense_table = table

    def dense_logits(self, h: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """The naive S @ K.T head, with the same beta/unigram terms applied."""
        flat = h.reshape(-1, self.d_model)
        S = self.project(flat)
        logits = S.to(table.dtype) @ table.t()
        logits = logits * self.beta[self.occ.lengths].to(logits.dtype)
        if self.unigram is not None:
            logits = logits + self.unigram.to(logits.dtype)
        return logits.view(*h.shape[:-1], self.V)


def analytic_op_counts(packed, d_model: int, D: int, N: int) -> dict:
    """Multiply-accumulate counts.  Not a wall-clock claim -- see E3."""
    dense = N * d_model * packed.V
    kas_proj = N * d_model * D
    kas_table = N * 256 * packed.L_max * (D // packed.L_max)
    kas_sum = N * packed.total_bytes
    kas = kas_proj + kas_table + kas_sum
    return {
        "dense_macs": dense,
        "kas_macs": kas,
        "kas_proj": kas_proj,
        "kas_table": kas_table,
        "kas_sum": kas_sum,
        # The strict ratio charges KAS for the d_model -> D projection, which
        # the dense head does not need.  The looser ratio counts only the
        # vocabulary-scaling part; it is the bigger number and the weaker claim.
        "ratio": dense / kas,
        "ratio_excluding_projection": dense / (kas_table + kas_sum),
    }
