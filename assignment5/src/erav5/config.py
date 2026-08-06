"""
ERA V5 — Mixture & Curriculum Plan
config.py — every input assumption in one place, so a reviewer can change
one number and re-run the whole plan.

Provenance of each figure is tagged:
  [V4]     measured on the ERA V4 run
  [SESS]   stated in session material
  [PUB]    published external figure (Sangraha, Stack-v2, FineWeb, etc.)
  [EST]    our estimate — the numbers a reviewer should push hardest on
"""

# ---------------------------------------------------------------------------
# 1. MODEL FAMILY
# ---------------------------------------------------------------------------
# V4 reference: 1.15T tokens, 120B MoE, + 9B / 5B / 2B dense variants. [V4]
V4 = {
    "tokens": 1.15e12,
    "total_params": 120e9,
    "shards": 33_353,
    "gpu_days": 67 * 8,          # 67 days on an 8-GPU node [V4]
    "usd": 100_000,
}

MODELS = {
    # name          total     active   role
    "V5-Proxy-1B":  dict(total=1.0e9,   active=1.0e9,   kind="dense", role="mixture ablation"),
    "V5-Proxy-3B":  dict(total=3.0e9,   active=3.0e9,   kind="dense", role="mixture ablation"),
    "V5-Seed-8B":   dict(total=8.0e9,   active=8.0e9,   kind="dense", role="growth seed / data gate"),
    "V5-Base":      dict(total=120e9,   active=15e9,    kind="moe",   role="V4-parity flagship"),
    "V5-Max":       dict(total=200e9,   active=20e9,    kind="moe",   role="stretch, if collection over-delivers"),
}
FLAGSHIP = "V5-Base"

# ---------------------------------------------------------------------------
# 2. COMPUTE ENVELOPE  (change these to match your actual cluster)
# ---------------------------------------------------------------------------
CLUSTER = {
    "n_gpus":            256,      # [EST] target cluster for V5
    "gpu_peak_tflops":   989.0,    # H100 BF16 dense peak
    "mfu":               0.40,     # [EST] realistic MoE MFU
    "days":              90,       # [EST] wall-clock training window
    "precision_speedup": 1.0,      # 2.0 if FP8 training
}

# Scaling-law constants
CHINCHILLA_TOK_PER_PARAM = 20.0        # [SESS] compute-optimal floor
OVERTRAIN_TARGET         = 267.0       # [EST] tok/active-param — serving-optimised
REPETITION_FREE_EPOCHS   = 4.0         # [SESS] repeats ~free to ~4 passes
REPETITION_DEAD_EPOCHS   = 16.0        # [SESS] wasted compute by ~16 passes

# ---------------------------------------------------------------------------
# 3. TOKEN BUDGETS
# ---------------------------------------------------------------------------
# Collection target is deliberately ~2x the training budget so the OPUS-class
# selector can be *selective* instead of being forced into repetition.
COLLECTION_TARGET = 8.0e12    # [EST] cleaned, gated, provenance-stamped
TRAIN_BUDGET      = 4.0e12    # [EST] tokens actually consumed by V5-Base
ANNEAL_RESERVE    = 0.20e12   # held back from TRAIN_BUDGET for the cooldown

# ---------------------------------------------------------------------------
# 4. DATASET INVENTORY
# ---------------------------------------------------------------------------
# unique_tokens = post-clean, post-global-dedup, license-cleared UNIQUE supply.
# loss_frac     = fraction of tokens that carry gradient (Session 5 loss-mapping).
#                 Agentic trajectories are mostly tool OUTPUT = context-only.
# tier          = A (scarce/precious, anneal-eligible) / B (bulk) / C (filler)
INVENTORY = [
    # ---- English -----------------------------------------------------------
    dict(name="FineWeb-Edu / DCLM-grade web", lane="english", unique=3.20e12,
         loss_frac=1.00, tier="B", license="permissive", src="[PUB]"),
    dict(name=".in news / gov / courts / Hansard", lane="english", unique=0.18e12,
         loss_frac=1.00, tier="A", license="mixed", src="[EST]"),
    dict(name="Books / long-form English", lane="english", unique=0.35e12,
         loss_frac=1.00, tier="A", license="mixed", src="[EST]"),
    dict(name="Wikipedia + reference", lane="english", unique=0.05e12,
         loss_frac=1.00, tier="A", license="share-alike", src="[PUB]"),

    # ---- Code --------------------------------------------------------------
    dict(name="Stack-v2 permissive (deduped, lint/exec-gated)", lane="code", unique=0.78e12,
         loss_frac=1.00, tier="B", license="permissive", src="[PUB]"),
    dict(name="PRs / issues / commit histories", lane="code", unique=0.09e12,
         loss_frac=1.00, tier="A", license="permissive", src="[EST]"),
    dict(name="Notebooks / competitive programming", lane="code", unique=0.04e12,
         loss_frac=1.00, tier="A", license="mixed", src="[EST]"),

    # ---- Math & science ----------------------------------------------------
    dict(name="FineMath-grade + OpenWebMath", lane="math", unique=0.11e12,
         loss_frac=1.00, tier="A", license="permissive", src="[PUB]"),
    dict(name="arXiv / PMC full text", lane="math", unique=0.10e12,
         loss_frac=1.00, tier="B", license="mixed", src="[PUB]"),
    dict(name="NCERT / JEE / UPSC prep (en+hi+regional)", lane="math", unique=0.04e12,
         loss_frac=1.00, tier="A", license="unknown", src="[EST]"),

    # ---- Indic: VERIFIED NATIVE  (the scarcest thing we own) ---------------
    # Sangraha headline 251B is ~65% MT; verified public native < 100B. [PUB]
    dict(name="Sangraha-verified native core", lane="indic_verified", unique=0.042e12,
         loss_frac=1.00, tier="A", license="mixed", src="[PUB]"),
    dict(name="OCR'd books/news, conf-gated + native audit", lane="indic_verified", unique=0.030e12,
         loss_frac=1.00, tier="A", license="mixed", src="[EST]"),
    dict(name="Legacy-font rescue (KrutiDev/Shree-Lipi/Bamini)", lane="indic_verified", unique=0.018e12,
         loss_frac=1.00, tier="A", license="mixed", src="[EST]"),

    # ---- Indic: UNVERIFIED NATIVE -----------------------------------------
    dict(name="FineWeb2-Indic + broader Sangraha pool", lane="indic_unverified", unique=0.28e12,
         loss_frac=1.00, tier="B", license="permissive", src="[PUB]"),
    dict(name="Parliament / judgments / gov records", lane="indic_unverified", unique=0.09e12,
         loss_frac=1.00, tier="A", license="gov-open", src="[EST]"),
    dict(name="ASR-mined speech transcripts", lane="indic_unverified", unique=0.08e12,
         loss_frac=1.00, tier="B", license="mixed", src="[EST]"),

    # ---- Indic: TRANSLATED -------------------------------------------------
    dict(name="BPCC + Samanantar parallel corpus", lane="indic_translated", unique=0.06e12,
         loss_frac=1.00, tier="B", license="permissive", src="[PUB]"),
    dict(name="Sangraha MT-flagged portion", lane="indic_translated", unique=0.16e12,
         loss_frac=1.00, tier="C", license="permissive", src="[PUB]"),

    # ---- Indic: SYNTHETIC (generated on demand, QC-gated) ------------------
    dict(name="Structure-preserving translation (14 langs)", lane="indic_synthetic", unique=None,
         loss_frac=1.00, tier="B", license="own", src="[EST]"),
    dict(name="Transliteration doubles (script<->roman)", lane="indic_synthetic", unique=None,
         loss_frac=1.00, tier="B", license="own", src="[EST]"),
    dict(name="Distilled CoT re-rendered in Indic", lane="indic_synthetic", unique=None,
         loss_frac=1.00, tier="A", license="own", src="[EST]"),

    # ---- Agentic  (loss_frac is the headline trap) -------------------------
    dict(name="Public agent trajectories (ToolBench/WebArena-class)", lane="agentic",
         unique=0.050e12, loss_frac=0.35, tier="B", license="mixed", src="[PUB]"),
    dict(name="Own sandbox self-play (terminal/browser/Indian-stack)", lane="agentic",
         unique=None, loss_frac=0.38, tier="A", license="own", src="[EST]"),

    # ---- Reasoning ---------------------------------------------------------
    dict(name="Public worked-solution / CoT corpora", lane="reasoning", unique=0.030e12,
         loss_frac=1.00, tier="A", license="mixed", src="[PUB]"),
    dict(name="RLVR-verified distilled long CoT", lane="reasoning", unique=None,
         loss_frac=1.00, tier="A", license="own", src="[EST]"),

    # ---- Forums & code-mix -------------------------------------------------
    dict(name="Forums / romanized social (Hinglish, Tanglish)", lane="forums", unique=0.35e12,
         loss_frac=1.00, tier="B", license="mixed", src="[EST]"),

    # ---- Other multilingual ------------------------------------------------
    dict(name="CulturaX / CC multilingual (non-Indic)", lane="multilingual", unique=1.50e12,
         loss_frac=1.00, tier="C", license="permissive", src="[PUB]"),
]

# ---------------------------------------------------------------------------
# 5. TARGET MIXTURE (global shares of TRAIN_BUDGET)
# ---------------------------------------------------------------------------
MIXTURE = {
    "english":            0.30,
    "code":               0.22,
    "math":               0.12,
    "indic_verified":     0.045,
    "indic_unverified":   0.070,
    "indic_translated":   0.045,
    "indic_synthetic":    0.040,
    "agentic":            0.060,
    "reasoning":          0.040,
    "forums":             0.030,
    "multilingual":       0.030,
}

INDIC_LANES = ["indic_verified", "indic_unverified", "indic_translated", "indic_synthetic"]
NATIVE_INDIC_LANES = ["indic_verified", "indic_unverified"]

# Max epochs we are willing to run per lane (repetition discipline)
MAX_EPOCHS = {
    "english": 1.0, "code": 1.5, "math": 2.0,
    "indic_verified": 2.0,      # scarce + eval-adjacent -> hard cap
    "indic_unverified": 1.5,
    "indic_translated": 1.0,
    "indic_synthetic": 1.0,
    "agentic": 1.5, "reasoning": 1.5,
    "forums": 1.0, "multilingual": 1.0,
}

# Protected always-on floor. V4 used 8% Indic. [V4/SESS]
# We tighten the DEFINITION: only NATIVE Indic counts toward the floor.
FLOOR = dict(pct=0.08, lanes=NATIVE_INDIC_LANES, batch_seqs=512)

# Synthetic discipline: synthetic Indic must not exceed native Indic. [SESS 30-50% band]
SYNTHETIC_PARITY_CAP = 1.0

# ---------------------------------------------------------------------------
# 6. FERTILITY (from the Session-3 assignment, carried forward unchanged)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 196_608
FERTILITY_TARGETS = {
    "english": 1.40, "hinglish": 1.45, "hindi": 1.55, "punjabi": 1.65,
    "gujarati": 1.70, "urdu": 1.70, "bengali": 1.75, "marathi": 1.80,
    "odia": 1.80, "assamese": 1.80, "telugu": 1.95, "kannada": 1.95,
    "tamil": 2.05, "malayalam": 2.15,
}
# Approximate share of Indian-language internet users. [EST]
LANG_WEIGHTS = {
    "hindi": 0.335, "bengali": 0.092, "marathi": 0.083, "telugu": 0.077,
    "tamil": 0.070, "gujarati": 0.056, "urdu": 0.050, "kannada": 0.046,
    "odia": 0.036, "malayalam": 0.035, "punjabi": 0.035, "assamese": 0.016,
    "hinglish": 0.069,
}
BASELINE_FERTILITY = {"english_centric_bpe": 4.5, "gemma4_262k": 2.2}
