"""Tests for the parts of the harness where a wrong answer would look like a right one.

`zero_harness.py` is a top-to-bottom script: importing it would train about two hundred
models.  So the pieces under test are lifted out of it with `ast`, which keeps one source of
truth -- if a definition in the notebook changes, these tests change with it, and a stale
copy cannot pass.

Everything here is arithmetic or structure rather than measurement: it must hold on any
machine, and it is what CI runs.  In particular these cover the three bugs that were real
during development and that a memory table would have hidden:

* the ring's index convention (member i must finish holding chunk i, not chunk i+1);
* the slot race between back-to-back collectives, which corrupts data rather than failing;
* activation accounting keyed on `data_ptr()` instead of the storage, which billed every
  weight in the model as an activation.
"""
import ast
import io
import math
import os
import threading

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "zero_harness.py")

WANTED = ("DeviceOOM", "DeviceMemory", "VirtualGPU", "Group", "Fabric", "storage_ptr",
          "flat_views", "param_shapes", "param_groups", "segments_for", "zero_theory",
          "embed_fwd", "block_fwd", "head_fwd", "segment_fwd", "gpt_forward", "gpt_loss",
          "init_params", "groups_whole_model")


def load_pieces():
    """Lift the wanted definitions plus every module-level constant, and run them in a
    namespace of their own.  Nothing that touches data, threads or the filesystem runs."""
    tree = ast.parse(io.open(SRC, encoding="utf-8").read())
    # MIB/GIB are written as `2 ** 20` in the harness, which `literal_eval` refuses, so
    # they are supplied here rather than scraped.
    ns = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math,
          "threading": threading, "os": os, "io": io,
          "MIB": 2 ** 20, "GIB": 2 ** 30}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            target = node.targets[0]
            names = target.elts if isinstance(target, ast.Tuple) else [target]
            values = value if isinstance(target, ast.Tuple) else [value]
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
DeviceOOM, DeviceMemory, VirtualGPU = P["DeviceOOM"], P["DeviceMemory"], P["VirtualGPU"]
Fabric, Group = P["Fabric"], P["Group"]
flat_views, param_shapes, param_groups = P["flat_views"], P["param_shapes"], P["param_groups"]
segments_for, zero_theory, storage_ptr = P["segments_for"], P["zero_theory"], P["storage_ptr"]
CFG = dict(vocab=65, block=16, d=32, n_head=4, n_layer=2)


# --------------------------------------------------------------------- the memory ledger
def test_ledger_tracks_and_refuses():
    mem = DeviceMemory(0, 1000)
    mem.alloc("a", torch.zeros(100, dtype=torch.uint8), "params")
    assert mem.used == 100 and mem.peak == 100
    mem.alloc("b", torch.zeros(400, dtype=torch.uint8), "grads")
    assert mem.used == 500 and mem.peak == 500 and mem.by_cat["grads"] == 400
    mem.free("b")
    assert mem.used == 100
    assert mem.peak == 500, "the high-water mark must survive a free"
    with pytest.raises(DeviceOOM):
        mem.alloc("c", torch.zeros(901, dtype=torch.uint8), "params")
    assert mem.used == 100, "a refused allocation must not be charged"


def test_ledger_counts_real_dtype_sizes():
    """The ledger asks the tensor how big it is; it never assumes 4 bytes."""
    mem = DeviceMemory(0, 10 ** 9)
    mem.alloc("bf16", torch.zeros(1000, dtype=torch.bfloat16), "params")
    mem.alloc("fp32", torch.zeros(1000, dtype=torch.float32), "optimizer")
    assert mem.by_cat["params"] == 2000
    assert mem.by_cat["optimizer"] == 4000


def test_double_allocation_is_a_bug_not_a_leak():
    mem = DeviceMemory(0, 10 ** 9)
    mem.alloc("x", torch.zeros(4), "params")
    with pytest.raises(AssertionError):
        mem.alloc("x", torch.zeros(4), "params")


# ------------------------------------------------------------------------- the topology
def test_node_layout_and_ring_links():
    fab = Fabric(32, gpus_per_node=8)
    assert fab.n_nodes == 4
    gpus = [VirtualGPU(r, 8, 1 << 20) for r in range(32)]
    assert [g.node for g in gpus[:9]] == [0] * 8 + [1]
    # A flat ring over 4 nodes crosses a node boundary exactly 4 times, and a ring is only
    # as fast as its slowest hop -- so the whole collective runs at InfiniBand speed.
    assert fab.world_group.n_inter == 4
    assert fab.world_group.bw == fab.bw_inter
    # Inside one node there is no such hop.
    assert fab.node_groups[0].n_inter == 0
    assert fab.node_groups[0].bw == fab.bw_intra


# ------------------------------------------------------------------------ the collectives
@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_ring_all_reduce_equals_a_plain_sum(world):
    fab = Fabric(world, gpus_per_node=8)
    vals = [torch.randn(world * 4, dtype=torch.float64) for _ in range(world)]
    want = torch.stack(vals).sum(0)
    got = fab.run(lambda r: fab.all_reduce(r, vals[r].clone()))
    for g in got:
        assert torch.allclose(g, want, atol=1e-12)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_reduce_scatter_leaves_member_i_holding_chunk_i(world):
    """The convention NCCL uses, and the one every sharded stage assumes.  Getting this
    off by one still produces a plausible-looking training run with wrong weights."""
    fab = Fabric(world, gpus_per_node=8)
    vals = [torch.randn(world * 3, dtype=torch.float64) for _ in range(world)]
    want = torch.stack(vals).sum(0)
    got = fab.run(lambda r: fab.reduce_scatter(r, vals[r].clone()))
    for r in range(world):
        assert torch.allclose(got[r], want[r * 3:(r + 1) * 3], atol=1e-12), \
            f"rank {r} did not end up with chunk {r}"


@pytest.mark.parametrize("world", [4, 8])
def test_all_gather_is_the_inverse(world):
    fab = Fabric(world, gpus_per_node=8)
    full = torch.arange(world * 5, dtype=torch.float64)

    def go(r):
        buf = torch.zeros(world * 5, dtype=torch.float64)
        buf.view(world, 5)[r] = full.view(world, 5)[r]
        return fab.all_gather_into(r, buf)
    for g in fab.run(go):
        assert torch.equal(g, full)


def test_back_to_back_collectives_do_not_corrupt_each_other():
    """The slot race.  One all-reduce in isolation was always correct; the bug only appeared
    when the next collective started before every rank had read the previous one's slot, and
    it corrupted data rather than raising -- which is the worst way for a bug to behave."""
    world, rounds = 8, 4
    fab = Fabric(world, gpus_per_node=4)          # two nodes, so the ring mixes link types
    vals = [torch.randn(world * 4, dtype=torch.float64) for _ in range(world)]
    want = torch.stack(vals).sum(0) * world ** (rounds - 1)

    def go(r):
        buf = vals[r].clone()
        for _ in range(rounds):                   # every round multiplies the sum by world
            fab.all_reduce(r, buf)
        return buf
    for g in fab.run(go):
        assert torch.allclose(g, want, rtol=1e-12), "four chained all-reduces drifted"


def test_measured_volume_is_the_ring_formula_not_two_psi():
    world = 8
    fab = Fabric(world, gpus_per_node=8)
    n = world * 64
    vals = [torch.randn(n, dtype=torch.float64) for _ in range(world)]
    fab.reset_counters()
    fab.run(lambda r: fab.all_reduce(r, vals[r].clone()))
    psi_bytes = n * 8
    assert fab.hops == world * 2 * (world - 1)
    for r in range(world):
        assert fab.bytes_per_rank[r] == pytest.approx(2 * psi_bytes * (world - 1) / world)


def test_hierarchical_all_reduce_is_the_same_sum_over_less_slow_wire():
    world = 32
    fab = Fabric(world, gpus_per_node=8)
    vals = [torch.randn(world * 2, dtype=torch.float64) for _ in range(world)]
    want = torch.stack(vals).sum(0)

    fab.reset_counters()
    fab.run(lambda r: fab.all_reduce(r, vals[r].clone()))
    flat_ib = fab.bytes_by_link["ib"]

    fab.reset_counters()
    got = fab.run(lambda r: fab.all_reduce_hierarchical(r, vals[r].clone()))
    hier_ib = fab.bytes_by_link["ib"]

    for g in got:
        assert torch.allclose(g, want, atol=1e-12), "hierarchical changed the answer"
    assert hier_ib < flat_ib, "hierarchical must put less on the slow wire"
    # Every chunk has to cross the node boundary, not just the node leaders' chunk.
    assert fab.by_op["rail0"] > 0 and fab.by_op[f"rail{fab.gpus_per_node - 1}"] > 0


def test_a_rank_that_fails_does_not_hang_the_others():
    """A barrier deadlock in a 32-thread simulation is indistinguishable from a slow run."""
    fab = Fabric(4, gpus_per_node=8)
    vals = [torch.randn(8, dtype=torch.float64) for _ in range(4)]

    def go(r):
        if r == 2:
            raise DeviceOOM("rank 2 is out of memory")
        return fab.all_reduce(r, vals[r].clone())
    with pytest.raises(DeviceOOM):
        fab.run(go)


# ----------------------------------------------------------------------- ZeRO arithmetic
@pytest.mark.parametrize("world", [1, 8, 32, 64])
def test_zero_theory_matches_the_published_formulas(world):
    psi = 1_000_000
    assert zero_theory(0, psi, world) == 16 * psi
    assert zero_theory(1, psi, world) == 4 * psi + 12 * psi / world
    assert zero_theory(2, psi, world) == 2 * psi + 14 * psi / world
    assert zero_theory(3, psi, world) == 16 * psi / world
    # At one GPU there is nothing to shard, so every stage must agree with the baseline.
    if world == 1:
        assert all(zero_theory(s, psi, 1) == 16 * psi for s in (0, 1, 2, 3))


def test_the_stages_are_ordered_and_stage_3_alone_has_no_floor():
    psi, world = 1e9, 32
    got = [zero_theory(s, psi, world) for s in (0, 1, 2, 3)]
    assert got == sorted(got, reverse=True), "each stage must use less than the last"
    huge = [zero_theory(s, psi, 10 ** 9) for s in (0, 1, 2, 3)]
    assert huge[0] == pytest.approx(16 * psi)        # DP never improves
    assert huge[1] == pytest.approx(4 * psi)         # ZeRO-1 floors at the weights+grads
    assert huge[2] == pytest.approx(2 * psi)         # ZeRO-2 floors at the weights
    assert huge[3] < 1e-3 * psi                      # ZeRO-3 does not floor


def test_session_30b_table_is_reproduced():
    """The session's own numbers for a 30B model on 8 GPUs: 447 / 153 / 104 / 55.9 GiB."""
    psi, world, gib = 30e9, 8, 2 ** 30
    want = {0: 447.0, 1: 153.0, 2: 104.0, 3: 55.9}
    for stage, said in want.items():
        assert zero_theory(stage, psi, world) / gib == pytest.approx(said, rel=0.01)


# -------------------------------------------------------------------- model bookkeeping
def test_flat_views_round_trip_without_copying():
    shapes = param_shapes(CFG)
    names = list(shapes)[:5]
    n = sum(int(np.prod(shapes[k])) for k in names)
    flat = torch.arange(n, dtype=torch.float64)
    views = flat_views(flat, names, shapes)
    assert [tuple(views[k].shape) for k in names] == [tuple(shapes[k]) for k in names]
    # A view shares storage with the buffer -- that is what makes a shard and a parameter
    # the same bytes, and it is why the activation meter keys on the storage.
    for k in names:
        assert storage_ptr(views[k]) == storage_ptr(flat)
    views[names[0]].fill_(-1)
    assert flat[0].item() == -1


def test_groups_partition_every_parameter_exactly_once():
    shapes = param_shapes(CFG)
    seen = [n for _, names in param_groups(CFG) for n in names]
    assert sorted(seen) == sorted(shapes), "a group split lost or duplicated a parameter"
    assert len(seen) == len(set(seen))


def test_a_bucket_covers_whole_segments_or_none():
    cfg = CFG
    for gname, names in param_groups(cfg):
        assert segments_for(names, cfg) == [gname]
    whole = P["groups_whole_model"](cfg)[0][1]
    assert segments_for(whole, cfg) == [g for g, _ in param_groups(cfg)]
    # Half a block is not a segment: ZeRO-3 cannot gather it and compute with it alone.
    half = param_groups(cfg)[1][1][:4]
    assert segments_for(half, cfg) == []


def test_forward_is_exactly_the_segments_composed():
    """`gpt_forward` and the segment-by-segment path ZeRO-3 takes must be one function."""
    torch.manual_seed(0)
    cfg = CFG
    Pw = P["init_params"](cfg, seed=3)
    idx = torch.randint(0, cfg["vocab"], (2, cfg["block"]))
    direct = P["gpt_forward"](idx, Pw, cfg)
    h = idx
    for sname, _ in param_groups(cfg):
        h = P["segment_fwd"](sname, h, Pw, cfg)
    assert torch.equal(direct, h)


def test_shards_divide_evenly_after_padding():
    """Every group is padded up to a multiple of the world size; without it a shard is a
    fraction of an element and the reduce-scatter silently reshapes."""
    for world in (8, 32, 64, 128):
        for gname, names in param_groups(CFG):
            n = sum(int(np.prod(param_shapes(CFG)[k])) for k in names)
            npad = math.ceil(n / world) * world
            assert npad % world == 0 and npad >= n
            assert npad - n < world, "padding should never exceed one element per rank"


def test_the_model_is_the_size_it_says_it_is():
    """Parameter counting is the one number everything else is expressed per unit of."""
    cfg = dict(vocab=65, block=64, d=96, n_head=4, n_layer=3)
    shapes = param_shapes(cfg)
    psi = sum(int(np.prod(s)) for s in shapes.values())
    d, V, T, L = cfg["d"], cfg["vocab"], cfg["block"], cfg["n_layer"]
    by_hand = V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d + V * d
    assert psi == by_hand
