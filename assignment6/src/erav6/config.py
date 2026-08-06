"""
config.py — every knob for the Session-6 execution demo in one place.

Session 5 answered "what should the mixture be". Session 6 answers "how is that
mixture actually delivered, recorded and reconstructed". So this file holds only
the *execution* parameters. The mixture itself is imported from Session 5 (see
plan.py) rather than retyped, because a Session-6 system that silently disagrees
with the Session-5 plan it claims to execute is exactly the failure this whole
assignment is about.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# .../schoolofai_subbu/assignment6/src/erav6/config.py -> parents[2] == assignment6
A6 = Path(__file__).resolve().parents[2]
REPO = A6.parent

CORPUS_DIR = A6 / "corpus"
TOKENIZER_PATH = A6 / "tokenizer" / "tokenizer.json"

# Regenerable intermediates. Shards are large-ish binary objects; the graded
# bundle gets their manifests, not their bytes.
WORK = A6 / "work"
SHARD_DIR = WORK / "shards"

# The graded output. Structure is fixed by the assignment.
ARTIFACTS = A6 / "submission_artifacts"
ART_MANIFESTS = ARTIFACTS / "manifests"
ART_LEDGERS = ARTIFACTS / "ledgers"
ART_CHECKPOINTS = ARTIFACTS / "checkpoints"
ART_LOG = ARTIFACTS / "run.log"
ART_EVIDENCE_JSON = ARTIFACTS / "evidence.json"
ART_EVIDENCE_MD = ARTIFACTS / "evidence.md"
ART_PERF = ARTIFACTS / "performance.json"

# ---------------------------------------------------------------------------
# 2. SEEDS
# ---------------------------------------------------------------------------
# DATA_SEED fixes the planned data stream. It is stored in every checkpoint and
# every ledger branch record: a resumed run that used a different data seed
# would produce a different stream, and the system must be able to say so.
DATA_SEED = 20260806
MODEL_SEED = 7

# ---------------------------------------------------------------------------
# 3. TOKENIZER CONTRACT
# ---------------------------------------------------------------------------
# A proxy tokenizer: byte-level BPE over the toy corpus. It stands in for the
# Session-3 tokenizer (vocab 196,608). What is NOT a proxy is the contract
# around it -- frozen at build time, content-hashed, and bound into every shard
# manifest, so a shard tokenized with a different tokenizer cannot be admitted.
VOCAB_SIZE = 512
TOKENIZER_NAME = "erav6-bytebpe"
TOKENIZER_VERSION = "1.0.0"

# Canonical special tokens. Ids are fixed by position and are part of the
# tokenizer hash, so they cannot drift.
SPECIAL_TOKENS = [
    "<|pad|>",        # 0 - never loss-bearing, never attended
    "<|bos|>",        # 1 - sequence start
    "<|eos|>",        # 2 - document boundary
    "<|user|>",       # 3 - role marker: human turn      (context only)
    "<|assistant|>",  # 4 - role marker: model turn      (loss-bearing)
    "<|think|>",      # 5 - role marker: reasoning       (loss-bearing)
    "<|tool_call|>",  # 6 - role marker: model tool call (loss-bearing)
    "<|tool_obs|>",   # 7 - role marker: tool OUTPUT     (context only)
    "<|unk|>",        # 8 - byte fallback guard
]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
ROLE_TOKEN_IDS = {
    "user": 3, "assistant": 4, "think": 5, "tool_call": 6, "tool_obs": 7,
}

# ---------------------------------------------------------------------------
# 4. LOSS POLICY  (Session 5 section 2.1, confirmed by the P5 proxy run)
# ---------------------------------------------------------------------------
# The single most consequential table in the file. `tool_obs` carries NO loss:
# training on tool responses teaches a model to hallucinate observations. The
# Session-5 P5 proxy measured this -- held-out fabrication 73.3% -> 0.0% -- so
# here it is executed rather than asserted.
ROLE_LOSS_POLICY = {
    "text": True,          # plain pretraining span
    "user": False,         # prompt / request  -> context only
    "tool_obs": False,     # environment output -> context only
    "think": True,         # model-generated reasoning
    "tool_call": True,     # model-generated call
    "assistant": True,     # model-generated answer
}
CONTEXT_ONLY_ROLES = tuple(r for r, v in ROLE_LOSS_POLICY.items() if not v)

# ---------------------------------------------------------------------------
# 5. PACKING POLICY PER LANE
# ---------------------------------------------------------------------------
# "The correct policy depends on the data type." Plain prose tolerates
# concatenation; structured samples do not. Lanes whose samples must not leak
# into each other through attention get structure_preserving, which emits
# per-segment position ids and a block-diagonal attention mask.
PACKING_POLICY = {
    "english": "concat_and_chop",
    "multilingual": "concat_and_chop",
    "forums": "concat_and_chop",
    "indic_unverified": "concat_and_chop",
    "indic_translated": "concat_and_chop",
    "indic_synthetic": "concat_and_chop",
    "indic_verified": "best_fit",          # scarce: waste no slot
    "code": "best_fit",                    # keep file/function boundaries
    "math": "greedy",
    "reasoning": "structure_preserving",   # a trace needs room to finish
    "agentic": "structure_preserving",     # tool order must survive packing
}
ALL_PACKING_POLICIES = [
    "pad_only", "concat_and_chop", "greedy", "best_fit",
    "structure_preserving", "long_context",
]
# A sample longer than this fraction of the window is routed to long_context.
LONG_CONTEXT_FRACTION = 0.75

# ---------------------------------------------------------------------------
# 6. ADMISSION GATE  (Session 4's contract, made executable)
# ---------------------------------------------------------------------------
ALLOWED_LICENSES = {"permissive", "gov-open", "share-alike", "own"}
BLOCKED_LICENSES = {"unknown", "noncommercial", "restricted"}
REQUIRED_MANIFEST_FIELDS = [
    "shard_id", "source_ids", "doc_ids", "tokenizer_hash", "token_count",
    "language", "script", "lane", "license_tier", "cleaning_pipeline_hash",
    "dedup_status", "contamination_status", "eval_overlap_status",
    "content_hash", "parent_shard_ids",
]
CLEANING_PIPELINE = {"name": "erav4-clean", "version": "4.2.1"}

# Contamination fingerprints are TOKEN n-grams, not word n-grams. Tokens are
# what actually enter a batch, so a token-level fingerprint measures the thing
# the firewall is protecting. 24 tokens is roughly a 12-14 word run at this
# tokenizer's fertility -- long enough that innocent collisions are vanishingly
# unlikely, short enough to catch a quoted benchmark item.
CONTAMINATION_NGRAM = 24

# Canary strings live only in the eval suites. Finding one inside a training
# shard means the firewall has already been breached somewhere upstream.
CANARY_MARKERS = ["CANARY-ERAV5-"]

# ---------------------------------------------------------------------------
# 7. EXECUTION PROFILES
# ---------------------------------------------------------------------------
# The assignment is explicit that scale is not the point, so the default
# profile is sized to run in ~1-2 minutes on CPU while still exercising every
# subsystem: 5 curriculum stages, stage transitions with ramps, lane epochs
# rolling over, OPUS deferral and re-draw, two checkpoints before the crash.
PROFILES = {
    "demo": dict(
        # 256 rather than 128 because the corpus averages 167 tokens per
        # document: at 128 every policy would degenerate into "split the
        # document" and greedy/best-fit would have nothing to pack.
        seq_len=256,
        d_model=128, n_layer=4, n_head=4, d_ff=512, dropout=0.0,
        n_ranks=2, microbatch=2, grad_accum=2,      # global batch = 8 sequences
        total_steps=24,
        ckpt_every=6,
        crash_at_step=16,                            # after ckpt @12, before @18
        replay_interval=(6, 12),
        fork_from_step=6, fork_steps=4,
        opus_candidate_factor=2.0,                   # score 2x what we accept
        lr=3e-3, warmup_steps=4, grad_clip=1.0,
        token_trace_every=4,                         # full token trace cadence
        probe_tokens=256,                            # held-back loss-delta probe
    ),
    "full": dict(
        seq_len=256,
        d_model=192, n_layer=6, n_head=6, d_ff=768, dropout=0.0,
        n_ranks=2, microbatch=4, grad_accum=2,       # global batch = 16
        total_steps=96,
        ckpt_every=12,
        crash_at_step=64,
        replay_interval=(24, 36),
        fork_from_step=24, fork_steps=12,
        opus_candidate_factor=2.0,
        lr=3e-3, warmup_steps=8, grad_clip=1.0,
        token_trace_every=8,
        probe_tokens=512,
    ),
}
DEFAULT_PROFILE = "demo"

# ---------------------------------------------------------------------------
# 7b. EPOCH CAPS AT TOY SCALE
# ---------------------------------------------------------------------------
# Session 5's per-lane epoch caps (english 1.0, indic_verified 2.0, ...) are
# stated against 4T tokens of real supply. This corpus is ~12.6K tokens, so a
# run of any useful length MUST re-read it -- applying the caps unscaled would
# put every lane in breach on step one and turn the scarcity report into noise.
#
# The caps are therefore scaled by a single factor that preserves their relative
# tightness. The lanes Session 5 identifies as tightest against supply are still
# the lanes that breach first, which is the property the scarcity mechanism is
# supposed to demonstrate. The unscaled Session-5 cap is carried alongside the
# scaled one in the compiled timeline, so the audit shows both.
# Chosen so the lanes Session 5 caps at 1.0 epoch -- the ones it identifies as
# having the least slack against supply -- are the ones that breach here, while
# the lanes it gives 1.5-2.0 stay inside their budget. That preserves the
# ORDERING of the plan's repetition discipline at a corpus four orders of
# magnitude smaller than the one it was written for.
EPOCH_CAP_SCALE = 4.6

# ---------------------------------------------------------------------------
# 7c. SCARCITY FALLBACKS AND THE ANNEAL RESERVE
# ---------------------------------------------------------------------------
# "If the Indic lane requests more verified native tokens than are available,
# the plan must specify whether to: repeat, generate, reduce the share, or move
# the share to a later stage." The answer has to be declared per lane BEFORE the
# run, otherwise the loader improvises and no one can audit what it did.
#
# These follow Session 5 section 8's stated responses:
#   repeat        repetition is safer than generation where no independent
#                 verifier exists for generated tokens (code: SWE-bench leakage)
#   synthesize    only for lanes with a real verification gate -- execution
#                 (agentic), answer-checking (reasoning), COMET+audit (indic
#                 synthetic)
#   reduce_share  lanes with large supply headroom in the plan; the share
#                 follows the supply, not the reverse
SCARCITY_FALLBACK = {
    "english": "reduce_share",
    "code": "repeat",
    "math": "repeat",
    "indic_verified": "repeat",          # to cap only, never beyond
    "indic_unverified": "repeat",        # most slack of any lane
    "indic_translated": "reduce_share",
    "indic_synthetic": "synthesize",
    "agentic": "synthesize",             # sandbox self-play, execution-gated
    "reasoning": "synthesize",           # RLVR-verified distillation
    "forums": "reduce_share",
    "multilingual": "reduce_share",
}

# "If the agentic lane contains scarce Tier A trajectories, the schedule must
# reserve them instead of allowing them to be exhausted early."  One shard per
# listed lane is held back and becomes drawable only in the anneal stage. The
# audit checks this directly: a reserved shard must have zero consumption events
# before S4 and at least one from S4 onward.
ANNEAL_RESERVE_LANES = ["indic_verified", "agentic", "reasoning"]
ANNEAL_STAGE_PREFIX = "S4"

# ---------------------------------------------------------------------------
# 8. OPUS SELECTOR
# ---------------------------------------------------------------------------
# Utility is measured from the live model, never sampled. See opus.py.
OPUS = dict(
    proxy_version="opus-proxy-1.0.0",
    accept_quantile=0.50,       # score above the candidate-pool median -> accept
    defer_quantile=0.25,        # between defer and accept -> deferred, re-scored later
    redundancy_weight=0.35,     # penalty for shards seen recently
    stage_fit_weight=0.25,      # bonus for lanes the current stage is asking for
    recent_window=8,            # steps of history used for the redundancy term
    defer_max_age=6,            # a deferred candidate must be re-scored within N steps
)

# ---------------------------------------------------------------------------
# 9. BRANCHES
# ---------------------------------------------------------------------------
MAIN_BRANCH = "main"
FORK_BRANCH = "fork-anneal-early"
RUN_ID = "erav6-run-0001"
