"""Training loop and arm construction.

Trap 7 lives here.  Arms differ in throughput by up to ~2x, so a per-arm
wall-clock cap silently compares them at *different token counts*.  Everything
below is token-matched: a calibration probe measures throughput, one budget
that every arm can afford is chosen, and it is then held fixed.  A wall-clock
cap is an abort, not a stopping rule.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .model import GPT, build_codec_for_arm
from .vocab import PackedVocab


def build_arm(
    arm: str,
    token_bytes: list[bytes],
    d_c: int = 32,
    L_max: int = 128,
    seed: int = 0,
    d_model: int = 384,
    n_layer: int = 6,
    n_head: int = 6,
    block_size: int = 256,
    dropout: float = 0.0,
    vocab_size: int | None = None,
):
    """Return (model, packed, codec) for one experimental arm."""
    torch.manual_seed(seed)
    codec = build_codec_for_arm(arm, d_c=d_c, L_max=L_max, seed=seed)
    pack_L = codec.L_max if codec is not None else L_max
    packed = PackedVocab(token_bytes, L_max=pack_L)
    model = GPT(
        arm,
        vocab_size or packed.V,
        packed,
        codec=codec,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        dropout=dropout,
    )
    return model, packed, codec


def lr_at(step: int, total: int, peak: float, warmup: int, floor_frac: float = 0.1):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _amp(device: str, enabled: bool):
    if device == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def throughput_probe(
    model,
    stream,
    device: str,
    batch_size: int,
    steps: int = 12,
    amp: bool = True,
    grad_accum: int = 1,
) -> float:
    """Measured tokens/second, warm-up excluded."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    block = stream.block_size
    for i in range(steps):
        if i == 4:  # discard warm-up and any lazy CUDA init
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        for _ in range(grad_accum):
            x, y = stream.batch(batch_size)
            xb = torch.from_numpy(x).to(device, non_blocking=True)
            yb = torch.from_numpy(y).to(device, non_blocking=True)
            with _amp(device, amp):
                _, loss = model(xb, yb)
            (loss / grad_accum).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return (steps - 4) * grad_accum * batch_size * block / dt


def train(
    model,
    train_stream,
    device: str,
    tokens: int,
    batch_size: int = 8,
    grad_accum: int = 1,
    lr: float = 6e-4,
    weight_decay: float = 0.1,
    warmup_frac: float = 0.02,
    grad_clip: float = 1.0,
    amp: bool = True,
    log_every: int = 200,
    eval_fn=None,
    eval_every: int | None = None,
    on_log=None,
) -> dict:
    model.to(device)
    model.train()
    block = train_stream.block_size
    tokens_per_step = batch_size * block * grad_accum
    total_steps = max(1, tokens // tokens_per_step)
    warmup = max(1, int(warmup_frac * total_steps))

    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )

    history: list[dict] = []
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(total_steps):
        cur_lr = lr_at(step, total_steps, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        # Accumulate the loss on-device.  Calling .item() every step forces a
        # sync, which serialises batch preparation against GPU compute -- it
        # cost ~2x wall-clock here, and it understates the reported tok/s.
        running = torch.zeros((), device=device)
        for _ in range(grad_accum):
            x, y = train_stream.batch(batch_size)
            xb = torch.from_numpy(x).to(device, non_blocking=True)
            yb = torch.from_numpy(y).to(device, non_blocking=True)
            with _amp(device, amp):
                _, loss = model(xb, yb)
            (loss / grad_accum).backward()
            running += loss.detach() / grad_accum
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % log_every == 0 or step == total_steps - 1:
            rec = {"step": step, "train_loss": running.item(), "lr": cur_lr}
            if eval_fn is not None and eval_every and (
                step % eval_every == 0 or step == total_steps - 1
            ):
                rec.update({f"val_{k}": v for k, v in eval_fn(model).items()})
            history.append(rec)
            if on_log is not None:
                on_log(rec)

    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        "steps": total_steps,
        "tokens": total_steps * tokens_per_step,
        "tokens_per_step": tokens_per_step,
        "wall_s": wall,
        "tokens_per_s": total_steps * tokens_per_step / wall,
        "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "history": history,
    }


def choose_token_budget(
    rates: dict[str, float], budget_min: float, tokens_per_step: int
) -> int:
    """One budget every arm can afford inside ``budget_min`` minutes each."""
    slowest = min(rates.values())
    raw = int(slowest * budget_min * 60)
    return max(tokens_per_step, (raw // tokens_per_step) * tokens_per_step)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
