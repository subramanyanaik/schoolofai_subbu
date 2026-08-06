# Evidence — ERA V5 Session 6 Training Data Execution System

**11 / 11 requirements pass.** Run `erav6-run-0001`, profile `demo`.

Every row is recomputed by `src/erav6/evidence.py` from the generated artefacts, in a separate process that never imports the trainer. The `Evidence` column names the files the result was derived from.

| Requirement | Result | Evidence | Key numbers |
|---|---|---|---|
| Tokenizer integrity | **PASS** | Manifest record | `n_manifests_checked=29`, `n_mismatched=0` |
| Shards, manifests and immutability | **PASS** | Manifest + recomputed content hash | `n_shards=29`, `n_admitted=25`, `n_blocked=4`, `n_content_hash_failures=0` |
| Evaluation firewall | **PASS** | Blocked-shard event | `n_blocked_shards=4`, `n_distinct_shards_consumed=24` |
| Packing correctness | **PASS** | Packed-batch report | `n_sequences_rebuilt=192`, `n_violations=0`, `overall_packing_utilization=0.748332` |
| Mixture compliance, floors and curriculum | **PASS** | Planned versus actual shares | `max_abs_unexplained_drift=0.0`, `max_abs_drift_vs_planned=0.114583`, `total_sequences_substituted=26`, `floor_violations=0` |
| OPUS audit trail | **PASS** | Candidate decision records | `n_decisions=649`, `n_deferred=154`, `drill_n_overrides=1`, `floor_override_demonstrated_in_drill=True` |
| Learning trace | **PASS** | Loss linked to source data | `n_shard_exposures=212`, `n_token_records=336` |
| Crash recovery | **PASS** | Expected and resumed batch ids | `expected_next_step=12`, `actual_next_step=12`, `n_hash_mismatches=0` |
| Replay and fork | **PASS** | Original and replay hashes | `n_microbatches_replayed=24`, `n_token_spans_verified=72`, `n_replay_mismatches=0` |
| Throughput and packing efficiency | **PASS** | Performance report | `useful_tokens_per_s=1706.507`, `rates_reconstructible=True` |
| End-to-end execution | **PASS** | Execution log | `effective_steps=24`, `n_markers=31`, `ledger_chains_ok=True` |

---

## How each result was derived

### Tokenizer integrity — PASS

recomputed sha256 of tokenizer.json and compared against the tokenizer_hash recorded in every shard manifest; verified a multi-byte round trip.

Files read: `manifests/_index.json`, `manifests/shard-*.json`, `tokenizer/tokenizer.json`

| Field | Value |
|---|---|
| `recomputed_tokenizer_hash` | b026dbabc4a964a630ff590c8d1126b5e6b71ed07d9c2b48c4afca87f5f376fb |
| `manifest_index_hash` | b026dbabc4a964a630ff590c8d1126b5e6b71ed07d9c2b48c4afca87f5f376fb |
| `n_manifests_checked` | 29 |
| `n_mismatched` | 0 |
| `vocab_size` | 512 |
| `multibyte_round_trip` | True |

### Shards, manifests and immutability — PASS

recomputed each shard's content hash from the bytes on disk and its index, compared against the manifest, and confirmed every manifest carries all required fields; the run separately demonstrated that a mutated shard fails this check.

Files read: `manifests/*.json`, `work/shards/*`

| Field | Value |
|---|---|
| `n_shards` | 29 |
| `n_admitted` | 25 |
| `n_blocked` | 4 |
| `n_content_hash_failures` | 0 |
| `n_incomplete_manifests` | 0 |
| `tamper_detection_demonstrated` | True |

### Evaluation firewall — PASS

walked every consume event on both branches, collected the shard ids that actually entered a gradient-bearing batch, and intersected them with the never-train registry; confirmed the gate blocked both a registered test shard and a training shard that shares an n-gram with a benchmark item.

Files read: `ledgers/consumption_*.jsonl`, `manifests/_index.json`, `manifests/_eval_registry.json`

| Field | Value |
|---|---|
| `fork_branch_clean` | True |
| `n_blocked_shards` | 4 |
| `n_registry_entries` | 6 |
| `n_contamination_fingerprints` | 266 |
| `n_distinct_shards_consumed` | 24 |

### Packing correctness — PASS

rebuilt every packed window from its ledger coordinates and re-tested five invariants on the rebuilt arrays: label shift, no loss on padding, no loss across an isolation boundary, position ids restarting per segment, and loss-mask hash equality.

Files read: `ledgers/consumption_main.jsonl`, `manifests/shard-*.json`

| Field | Value |
|---|---|
| `n_sequences_rebuilt` | 192 |
| `n_violations` | 0 |
| `overall_packing_utilization` | 0.748332 |
| `total_loss_bearing_tokens` | 34439 |

### Mixture compliance, floors and curriculum — PASS

recomputed realised lane shares from the consumption ledger and compared them against the quotas the timeline compiler produced from the Session-5 plan, requiring every deviation to be attributable to a logged scarcity substitution; re-checked the protected floor in every single batch rather than on average; verified anneal-reserved shards were untouched before S4.

Files read: `ledgers/consumption_main.jsonl`, `manifests/_index.json`

| Field | Value |
|---|---|
| `max_abs_unexplained_drift` | 0.0 |
| `max_abs_drift_vs_planned` | 0.114583 |
| `n_substitution_events` | 16 |
| `total_sequences_substituted` | 26 |
| `tolerance` | 0.02 |
| `floor_ok` | True |
| `floor_pct` | 0.08 |
| `floor_violations` | 0 |
| `min_floor_share` | 0.125 |
| `mean_floor_share` | 0.166667 |

### OPUS audit trail — PASS

read every decision record from the OPUS ledger and confirmed each candidate carries a status, a proxy score measured from the live model, and a reason; checked that deferred candidates are re-offered rather than lost; confirmed the protected-floor override fires in the controlled drill where the proxy is the English-tuned kind Session 5 warns about.

Files read: `ledgers/opus_main.jsonl`, `ledgers/_floor_drill.json`

| Field | Value |
|---|---|
| `n_decisions` | 649 |
| `n_floor_overrides_main_run` | 0 |
| `floor_override_demonstrated_in_drill` | True |
| `drill_unprotected_native_indic` | 0 |
| `drill_protected_native_indic` | 1 |
| `drill_n_overrides` | 1 |
| `drill_proxy_version` | opus-proxy-1.0.0-lane-biased-drill |
| `n_deferred` | 154 |
| `n_deferred_returned` | 162 |
| `every_candidate_has_a_decision` | True |

### Learning trace — PASS

confirmed every shard exposure in the learning ledger carries a measured probe loss before and after the optimizer step, linked to the shard and lane it came from, and that every token-level record carries its own cross-entropy and perplexity linked back to a source document and shard.

Files read: `ledgers/learning_main.jsonl`, `ledgers/token_trace_main.jsonl`

| Field | Value |
|---|---|
| `n_shard_exposures` | 212 |
| `n_exposures_with_probe_delta` | 212 |
| `n_token_trace_batches` | 7 |
| `n_token_records` | 336 |
| `n_token_records_linked_to_source` | 336 |
| `all_exposures_linked_to_shard` | True |

### Crash recovery — PASS

split the ledger at the rollback marker, compared the batch content hash the crashed process recorded at each step against the one the resumed process produced, and confirmed the first resumed step is exactly the step the checkpoint named -- no skip, no repeat; re-verified the checkpoint's ledger-head binding.

Files read: `ledgers/consumption_main.jsonl`, `checkpoints/*.json`, `run.log`

| Field | Value |
|---|---|
| `crash_recorded` | True |
| `crash_exit_code` | 137 |
| `crash_at_step` | 16 |
| `rollback_recorded` | True |
| `rollback_to_offset` | 86 |
| `expected_next_step` | 12 |
| `actual_next_step` | 12 |
| `next_batch_matched` | True |
| `n_steps_compared` | 4 |
| `n_hash_mismatches` | 0 |

### Replay and fork — PASS

re-opened the immutable shards, re-cut every recorded token span, rebuilt each window from its ledger coordinates, and compared all three things the assignment asks a replay to prove: the batch id (recomputed from branch/step/rank/microbatch), the token spans (rebuilt from coordinates) and the hashes (loss-mask and full content); separately confirmed the fork recorded its divergence point and produced a genuinely different stream.

Files read: `ledgers/consumption_main.jsonl`, `ledgers/consumption_fork-anneal-early.jsonl`

| Field | Value |
|---|---|
| `n_microbatches_replayed` | 24 |
| `n_batch_ids_verified` | 24 |
| `n_token_spans_verified` | 72 |
| `n_replay_mismatches` | 0 |
| `fork_branch` | fork-anneal-early |
| `fork_parent_step` | 6 |
| `fork_divergence_recorded` | True |
| `fork_shared_steps` | 4 |
| `fork_diverging_steps` | 4 |

### Throughput and packing efficiency — PASS

re-divided the raw counters in performance.json to reproduce every headline rate, and cross-checked the reported packing utilization against the token counts independently recorded in the consumption ledger.

Files read: `performance.json`, `ledgers/consumption_main.jsonl`

| Field | Value |
|---|---|
| `wall_clock_s` | 23.521148 |
| `rates_reconstructible` | True |
| `reported_packing_utilization` | 0.744559 |
| `packing_utilization_from_ledger` | 0.748332 |
| `utilization_agrees` | True |
| `useful_tokens_per_s` | 1706.507 |
| `raw_tokens_per_s` | 2437.976 |
| `padding_waste` | 0.255441 |
| `context_only_fraction` | 0.059889 |
| `opus_acceptance_rate` | 0.345946 |

### End-to-end execution — PASS

confirmed run.log contains every required event name, that no check emitted a FAIL marker, that all ten ledger hash chains verify from genesis, and that the effective consumption ledger holds exactly one optimizer step per planned step with no gaps.

Files read: `run.log`, `ledgers/*.jsonl`

| Field | Value |
|---|---|
| `n_markers` | 31 |
| `ledger_chains_ok` | True |
| `effective_steps` | 24 |
| `expected_steps` | 24 |
| `steps_contiguous` | True |

---

## Execution log markers

| Marker | Result | Phase |
|---|---|---|
| `tokenizer_hash_verified` | PASS | - |
| `manifests_complete` | PASS | - |
| `all_shards_bound_to_tokenizer` | PASS | - |
| `shard_immutability_enforced` | PASS | - |
| `eval_shard_blocked` | PASS | - |
| `license_gate_enforced` | PASS | - |
| `validation_read_allowed_gradient_denied` | PASS | - |
| `mixture_integrates_to_session5_plan` | PASS | - |
| `protected_floor_satisfiable` | PASS | - |
| `scarcity_fallbacks_declared` | PASS | - |
| `packing_isolation_enforced` | PASS | - |
| `checkpoint_saved` | PASS | - |
| `checkpoint_saved` | PASS | - |
| `crash_produced_nonzero_exit` | PASS | - |
| `checkpoint_ledger_binding_verified` | PASS | - |
| `checkpoint_saved` | PASS | - |
| `checkpoint_saved` | PASS | - |
| `resume_next_batch_matched` | PASS | - |
| `resume_no_skipped_or_repeated_batches` | PASS | - |
| `resume_batch_hashes_identical` | PASS | - |
| `replay_hash_matched` | PASS | - |
| `fork_parent_binding_verified` | PASS | - |
| `fork_divergence_recorded` | PASS | - |
| `protected_floor_override_demonstrated` | PASS | - |
| `ledger_chain_intact` | PASS | - |
| `no_eval_data_in_training` | PASS | - |
| `mixture_deviation_fully_attributed` | PASS | - |
| `protected_floor_held_every_batch` | PASS | - |
| `anneal_reserve_respected` | PASS | - |
| `opus_every_candidate_decided` | PASS | - |
| `throughput_measured` | PASS | - |

