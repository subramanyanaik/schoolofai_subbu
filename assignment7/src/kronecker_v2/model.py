"""A small GPT with five interchangeable input/output arms.

The input embedding is the *dual* of the KAS head, and it is built the same
way: never materialise a |V| x D table (412 MB in fp16 at the GPT-2 vocab and
D=4096).  Contract it against the projection first, then apply the occupancy
matrix once.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import KroneckerCodec
from .kas_head import KASHead, Occupancy

ARMS = (
    "a0_bpe_tied",
    "a1_v1_dense",
    "a2_kronf_kas",
    "a3_kronf_kas_tied",
    "a4_kas_unigram",
)


# --------------------------------------------------------------- transformer


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        shape = (B, T, self.n_head, C // self.n_head)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


# ------------------------------------------------------------ input embedding


class KroneckerEmbedding(nn.Module):
    """Input side: e_v = znorm(K_v) @ W_proj, computed without forming K.

    ``table()`` is provably equal to ``codec.encode(token_bytes) @ W_proj``
    (with optional z-normalisation), and ``test_table_equals_direct_encode``
    checks exactly that -- it is the dual of the head's exactness gate.
    """

    def __init__(
        self,
        codec: KroneckerCodec,
        packed,
        d_model: int,
        occ: Occupancy | None = None,
        znorm: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.d_c, self.L_max, self.D = codec.d_c, codec.L_max, codec.D
        self.d_model = d_model
        self.V = packed.V
        self.znorm = znorm

        self.W_proj = nn.Parameter(
            torch.randn(self.D, d_model, dtype=dtype) * (self.D ** -0.5)
        )
        self.register_buffer("G", torch.from_numpy(codec.G.copy()).to(dtype))
        self.register_buffer("Phi", torch.from_numpy(codec.Phi.copy()).to(dtype))
        self.occ = occ if occ is not None else Occupancy(packed, dtype=dtype)

        if znorm:
            # Closed form.  sum(K_v) = sum_p (1/sqrt(L)) * g_sum[b_p] * phi_row_sum[p],
            # which is itself an occupancy product; and sum(K_v^2) = ||K_v||^2 = 1
            # because every codec vector is unit norm.  So no encoding pass is
            # needed, and mu/sigma are exact rather than estimated.
            g_sum = torch.from_numpy(codec.G.sum(axis=1)).to(torch.float64)
            phi_sum = torch.from_numpy(codec.Phi.sum(axis=1)).to(torch.float64)
            cell = (g_sum[:, None] * phi_sum[None, :]).reshape(-1, 1)
            occ64 = Occupancy(packed, dtype=torch.float64)
            mu = occ64.matmul(cell).squeeze(1) / self.D
            var = (1.0 / self.D) - mu * mu
            sigma = torch.sqrt(torch.clamp(var, min=1e-24))
            self.register_buffer("zn_scale", (1.0 / sigma).to(dtype))
            self.register_buffer("zn_shift", (-mu / sigma).to(dtype))

    def table(self) -> torch.Tensor:
        """(V, d_model).  One occupancy product, no |V| x D intermediate."""
        Wp = self.W_proj.view(self.d_c, self.L_max, self.d_model)
        # Fold the position basis into the projection: K = vec(M_pos @ Phi), so
        # <K, W_proj> = <M_pos, Phi-rotated W_proj>.
        Wpos = torch.einsum("pk,ckm->cpm", self.Phi.to(Wp.dtype), Wp)
        B = torch.einsum("vc,cpm->vpm", self.G.to(Wp.dtype), Wpos)
        out = self.occ.matmul(B.reshape(256 * self.L_max, self.d_model))
        if self.znorm:
            colsum = self.W_proj.sum(dim=0)
            out = out * self.zn_scale.to(out.dtype)[:, None] + (
                self.zn_shift.to(out.dtype)[:, None] * colsum[None, :]
            )
        return out

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.table())


# ------------------------------------------------------------------- the GPT


class GPT(nn.Module):
    def __init__(
        self,
        arm: str,
        vocab_size: int,
        packed,
        codec: KroneckerCodec | None = None,
        d_model: int = 384,
        n_layer: int = 6,
        n_head: int = 6,
        block_size: int = 256,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        self.arm = arm
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.wpe = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(
            [Block(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model, bias=False)

        # A single Occupancy is shared between input and output pathways.
        self.occ = Occupancy(packed, dtype=dtype) if codec is not None else None

        if arm == "a0_bpe_tied":
            self.wte = nn.Embedding(vocab_size, d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight  # tied
        elif arm == "a1_v1_dense":
            assert codec is not None and codec.variant == "v1"
            self.wte = KroneckerEmbedding(
                codec, packed, d_model, occ=self.occ, znorm=True, dtype=dtype
            )
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        else:
            assert codec is not None
            self.wte = KroneckerEmbedding(
                codec, packed, d_model, occ=self.occ, znorm=False, dtype=dtype
            )
            tied = self.wte.W_proj if arm == "a3_kronf_kas_tied" else None
            self.lm_head = KASHead(
                d_model,
                codec,
                packed,
                tied_weight=tied,
                dtype=dtype,
                unigram_bias=(arm == "a4_kas_unigram"),
                occ=self.occ,
            )

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -------------------------------------------------------------- forward

    def backbone(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"sequence {T} exceeds block size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.backbone(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1)
            )
        return logits, loss

    # ------------------------------------------------------------ accounting

    def n_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def param_breakdown(self) -> dict:
        seen = set()
        head, embed, body = 0, 0, 0
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if name.startswith("lm_head"):
                head += p.numel()
            elif name.startswith("wte"):
                embed += p.numel()
            else:
                body += p.numel()
        return {
            "total": head + embed + body,
            "head": head,
            "input_embedding": embed,
            "body": body,
        }

    def unused_parameters(self, device: torch.device | None = None) -> list[str]:
        """Every parameter must receive a gradient.  Trap 5 made this a test."""
        dev = device or next(self.parameters()).device
        idx = torch.randint(0, self.vocab_size, (2, 8), device=dev)
        self.zero_grad(set_to_none=True)
        _, loss = self(idx, idx)
        loss.backward()
        seen, unused = set(), []
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0:
                unused.append(name)
        self.zero_grad(set_to_none=True)
        return unused


def build_codec_for_arm(arm: str, d_c: int = 32, L_max: int = 128, seed: int = 0):
    """The codec each arm uses.  a0 has none: it is the BPE baseline."""
    if arm == "a0_bpe_tied":
        return None
    if arm == "a1_v1_dense":
        return KroneckerCodec(variant="v1", L_max=16, seed=seed)
    return KroneckerCodec(d_c=d_c, L_max=L_max, seed=seed)
