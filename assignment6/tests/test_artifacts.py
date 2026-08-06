"""
End-to-end tests over the GENERATED bundle.

These verify the properties that only exist once the demo has run: crash
recovery, replay, fork divergence, the firewall sweep over real consumption,
and the internal consistency of evidence.json against the ledgers it claims to
summarise.

They skip cleanly if `submission_artifacts/` is absent, so the unit suite runs
on a fresh clone:

    python run_demo.py                              # produce the bundle
    python -m unittest discover -s tests -v         # then check it

The point of keeping these separate from `test_invariants.py` is that these
tests are a SECOND, independent reading of the artefacts -- they re-derive the
same conclusions the evidence bundle reports, without going through
`evidence.py`. If the generator and these tests disagree, one of them is wrong,
and that is worth knowing.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

A6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A6 / "src"))

from erav6 import audit as AU                     # noqa: E402
from erav6 import catalog as CATALOG              # noqa: E402
from erav6 import config as C                     # noqa: E402
from erav6 import mixture as MIX                  # noqa: E402
from erav6 import plan                            # noqa: E402
from erav6.ledger import open_ledgers             # noqa: E402

ART = C.ARTIFACTS
HAVE_ARTIFACTS = (ART / "evidence.json").exists() and (ART / "run.log").exists()
SKIP = "run `python run_demo.py` first to generate submission_artifacts/"


@unittest.skipUnless(HAVE_ARTIFACTS, SKIP)
class ArtifactTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = C.PROFILES[C.DEFAULT_PROFILE]
        cls.cat = CATALOG.build_catalog(log=None)
        seqs = (cls.profile["n_ranks"] * cls.profile["microbatch"]
                * cls.profile["grad_accum"])
        cls.timeline = MIX.compile_timeline(cls.cat, cls.profile["total_steps"],
                                            seqs, cls.profile["seq_len"])
        cls.main = open_ledgers(ART / "ledgers", C.RUN_ID, C.MAIN_BRANCH)
        cls.fork = open_ledgers(ART / "ledgers", C.RUN_ID, C.FORK_BRANCH)
        cls.evidence = json.loads((ART / "evidence.json").read_text(encoding="utf-8"))
        cls.perf = json.loads((ART / "performance.json").read_text(encoding="utf-8"))
        cls.log = (ART / "run.log").read_text(encoding="utf-8")


class TestBundleStructure(ArtifactTestCase):
    def test_required_files_exist(self):
        for rel in ("run.log", "evidence.json", "evidence.md", "performance.json"):
            self.assertTrue((ART / rel).exists(), rel)
        for d in ("manifests", "ledgers", "checkpoints"):
            self.assertTrue((ART / d).is_dir(), d)
            self.assertTrue(any((ART / d).iterdir()), f"{d} is empty")

    def test_run_log_contains_every_required_event(self):
        for event in ["shards_created", "manifests_validated", "eval_shard_blocked",
                      "mixture_compiled", "batches_packed", "opus_decisions_recorded",
                      "checkpoint_saved", "crash_simulated", "run_resumed",
                      "historical_stream_replayed", "branch_forked",
                      "audit_completed", "performance_measured"]:
            self.assertIn(event, self.log, event)

    def test_run_log_has_the_named_pass_markers(self):
        """The five markers the assignment spells out, verbatim."""
        for marker in ["[PASS] tokenizer_hash_verified",
                       "[PASS] eval_shard_blocked",
                       "[PASS] checkpoint_saved",
                       "[PASS] resume_next_batch_matched",
                       "[PASS] replay_hash_matched"]:
            self.assertIn(marker, self.log, marker)

    def test_no_failed_markers(self):
        self.assertNotIn("[FAIL]", self.log)


class TestLedgerIntegrity(ArtifactTestCase):
    def test_every_chain_verifies(self):
        for name, result in self.main.verify_all().items():
            self.assertTrue(result["ok"], f"main/{name}: {result}")
        for name, result in self.fork.verify_all().items():
            self.assertTrue(result["ok"], f"fork/{name}: {result}")

    def test_effective_stream_has_one_step_per_planned_step(self):
        steps = sorted({e["global_step"]
                        for e in self.main.consumption.events("optimizer_step")})
        self.assertEqual(steps, list(range(self.profile["total_steps"])))

    def test_each_step_records_every_microbatch(self):
        expected = self.profile["n_ranks"] * self.profile["grad_accum"]
        for step in range(self.profile["total_steps"]):
            evs = self.main.consumption.events_for_step(step)
            self.assertLessEqual(len(evs), expected, step)
            self.assertGreater(len(evs), 0, step)
            pairs = {(e["rank"], e["microbatch_id"]) for e in evs}
            self.assertEqual(len(pairs), len(evs), f"duplicate microbatch at {step}")

    def test_consume_events_carry_the_replay_coordinates(self):
        for ev in self.main.consumption.events("consume")[:20]:
            for rec in ev["sequences"]:
                self.assertTrue(rec["members"])
                for m in rec["members"]:
                    for k in ("shard_id", "shard_start", "length", "offset",
                              "segment_id"):
                        self.assertIn(k, m)
                self.assertIn(rec["policy"], C.ALL_PACKING_POLICIES)

    def test_every_event_names_its_tokenizer_and_loader_version(self):
        for ev in self.main.consumption.events("consume")[:20]:
            self.assertEqual(ev["tokenizer_hash"], self.cat.tokenizer_hash)
            self.assertTrue(ev["dataloader_version"])


class TestCrashRecovery(ArtifactTestCase):
    def test_the_crash_was_a_real_process_death(self):
        crashes = [e for e in self.main.consumption.all_events()
                   if e["type"] == "crash"]
        self.assertTrue(crashes)
        self.assertNotEqual(crashes[0]["exit_code"], 0)
        self.assertIn("exit_code=137", self.log)

    def test_rollback_was_appended_not_truncated(self):
        before, after, rb = AU.split_at_rollback(self.main.consumption)
        self.assertIsNotNone(rb)
        self.assertTrue(before, "the crashed process's record was destroyed")
        self.assertTrue(after)
        self.assertTrue(rb["orphaned_steps"])

    def test_resumed_batches_are_byte_identical_to_the_crashed_run(self):
        before, after, rb = AU.split_at_rollback(self.main.consumption)
        orig = {e["global_step"]: e["batch_content_hash"] for e in before}
        res = {e["global_step"]: e["batch_content_hash"] for e in after}
        overlap = sorted(set(orig) & set(res))
        self.assertTrue(overlap, "no overlapping steps to compare")
        for step in overlap:
            self.assertEqual(orig[step], res[step], f"step {step} differs after resume")

    def test_next_batch_is_exactly_the_expected_one(self):
        before, after, rb = AU.split_at_rollback(self.main.consumption)
        ckpt = None
        for p in sorted((ART / "checkpoints").glob(f"ckpt-{C.MAIN_BRANCH}-*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec["ledger_offset"] == rb["to_offset"]:
                ckpt = rec
        self.assertIsNotNone(ckpt)
        ver = AU.resume_verification(before, after, ckpt["global_step"])
        self.assertTrue(ver["next_batch_matched"])
        self.assertEqual(ver["skipped_steps"], [])
        self.assertEqual(ver["repeated_steps"], [])

    def test_every_checkpoint_is_bound_to_a_ledger_position(self):
        found = 0
        for p in sorted((ART / "checkpoints").glob(f"ckpt-{C.MAIN_BRANCH}-*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            for key in ("ledger_offset", "ledger_head_hash", "loader_state",
                        "data_seed", "tokenizer_hash", "catalog_set_hash"):
                self.assertIn(key, rec, p.name)
            self.assertEqual(
                self.main.consumption.hash_at(rec["ledger_offset"]),
                rec["ledger_head_hash"], p.name)
            found += 1
        self.assertGreater(found, 1)

    def test_a_tampered_checkpoint_record_is_rejected(self):
        from erav6 import checkpoint as CK
        p = sorted((ART / "checkpoints").glob(f"ckpt-{C.MAIN_BRANCH}-*.json"))[0]
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["ledger_offset"] = rec["ledger_offset"] + 1
        with self.assertRaises(CK.LedgerBindingError):
            CK.restore(ART / "checkpoints", rec, model=None, optimizer=None,
                       ledger=self.main.consumption)


class TestReplayAndFork(ArtifactTestCase):
    def test_replay_reproduces_every_hash_in_the_interval(self):
        lo, hi = self.profile["replay_interval"]
        rep = AU.replay_interval(self.cat, self.main.consumption, lo, hi,
                                 self.profile["seq_len"])
        self.assertTrue(rep["ok"], rep["mismatches"][:3])
        self.assertGreater(rep["n_microbatches_replayed"], 0)
        self.assertGreater(rep["n_token_spans_verified"], 0)

    def test_replay_covers_a_whole_optimizer_step(self):
        lo, hi = self.profile["replay_interval"]
        steps = {e["global_step"]
                 for e in self.main.consumption.events_in_step_range(lo, hi)}
        self.assertEqual(steps, set(range(lo, hi)))

    def test_fork_records_its_divergence_point(self):
        fk = AU.fork_verification(self.main.consumption, self.fork.consumption,
                                  self.profile["fork_from_step"])
        self.assertTrue(fk["ok"], fk)
        self.assertEqual(fk["parent_branch"], C.MAIN_BRANCH)
        self.assertEqual(fk["parent_step"], self.profile["fork_from_step"])
        self.assertTrue(fk["parent_ledger_hash"])

    def test_fork_actually_diverges(self):
        fk = AU.fork_verification(self.main.consumption, self.fork.consumption,
                                  self.profile["fork_from_step"])
        self.assertGreater(fk["n_shared_steps"], 0)
        self.assertEqual(fk["n_diverging_steps"], fk["n_shared_steps"],
                         "a fork that reproduces the parent stream is not a fork")


class TestFirewall(ArtifactTestCase):
    def test_no_never_train_shard_ever_reached_a_gradient(self):
        for led in (self.main.consumption, self.fork.consumption):
            sweep = AU.firewall_sweep(led, self.cat.registry, self.cat)
            self.assertEqual(sweep["never_train_shards_consumed"], [])
            self.assertEqual(sweep["non_training_split_shards_consumed"], [])
            self.assertEqual(sweep["blocked_shards_consumed"], [])

    def test_the_contaminated_shard_was_blocked_and_never_consumed(self):
        contaminated = [d.shard_id for d in self.cat.decisions
                        if "eval_overlap" in d.reasons]
        self.assertTrue(contaminated)
        consumed = set()
        for ev in self.main.consumption.events("consume"):
            consumed.update(ev["shard_ids"])
        for sid in contaminated:
            self.assertNotIn(sid, consumed)

    def test_blocked_shards_are_absent_from_the_drawable_pool(self):
        for sid in self.cat.blocked:
            self.assertNotIn(sid, [s for ids in self.cat.by_lane.values() for s in ids])
            with self.assertRaises(PermissionError):
                self.cat.shard(sid)


class TestMixtureAndFloor(ArtifactTestCase):
    def test_every_deviation_from_plan_is_attributable(self):
        comp = AU.mixture_compliance(self.main.consumption, self.timeline)
        self.assertLessEqual(comp["max_abs_unexplained_drift"], 0.02, comp["lanes"])

    def test_the_floor_held_in_every_single_batch(self):
        floor = AU.floor_compliance(self.main.consumption)
        self.assertTrue(floor["ok"], floor["violating_steps"])
        self.assertEqual(floor["n_violations"], 0)
        self.assertGreaterEqual(floor["min_delivered_share"], plan.FLOOR_PCT)

    def test_all_five_curriculum_stages_executed(self):
        stages = {e["curriculum_stage"]
                  for e in self.main.consumption.events("optimizer_step")}
        self.assertEqual(stages, set(plan.STAGE_NAMES))

    def test_anneal_reserved_shards_were_held_back(self):
        res = AU.anneal_reserve_compliance(self.main.consumption, self.timeline)
        self.assertTrue(res["ok"], res.get("leaked_shards"))
        self.assertTrue(res["reserved_shards"])
        self.assertEqual(res.get("leaked_shards", []), [])

    def test_every_lane_was_served_at_least_once(self):
        seen = set()
        for ev in self.main.consumption.events("consume"):
            for rec in ev["sequences"]:
                seen.add(rec["lane"])
        self.assertEqual(seen, set(plan.LANES))


class TestOpusTrail(ArtifactTestCase):
    def test_every_candidate_carries_a_decision_and_a_reason(self):
        trail = AU.opus_audit(self.main.opus)
        self.assertTrue(trail["every_candidate_has_a_decision"])
        self.assertGreater(trail["n_decisions"], 0)

    def test_all_four_outcomes_are_represented(self):
        trail = AU.opus_audit(self.main.opus)
        for status in ("accepted", "rejected", "deferred"):
            self.assertIn(status, trail["by_status"], status)
        drill = json.loads((ART / "ledgers" / "_floor_drill.json")
                           .read_text(encoding="utf-8"))
        self.assertGreater(drill["protected"]["n_floor_overrides"], 0)

    def test_deferred_candidates_come_back(self):
        trail = AU.opus_audit(self.main.opus)
        self.assertGreater(trail["n_deferred"], 0)
        self.assertGreater(trail["n_deferred_returned"], 0)

    def test_rejected_data_is_retained_with_reasons(self):
        rejected = [e for e in self.main.opus.events("opus_decision")
                    if e["status"] == "rejected"]
        self.assertTrue(rejected)
        for e in rejected:
            self.assertTrue(e["rejection_reason"])
            self.assertTrue(e["shard_ids"])

    def test_scores_are_measured_not_constant(self):
        scores = [e["opus_score"] for e in self.main.opus.events("opus_decision")]
        self.assertGreater(len(set(scores)), 10,
                           "a selector whose scores never vary is not measuring anything")

    def test_the_floor_drill_shows_what_the_floor_prevents(self):
        drill = json.loads((ART / "ledgers" / "_floor_drill.json")
                           .read_text(encoding="utf-8"))
        self.assertTrue(drill["floor_was_necessary"])
        self.assertLess(drill["unprotected"]["native_indic_sequences"],
                        drill["floor_required"])
        self.assertGreaterEqual(drill["protected"]["native_indic_sequences"],
                                drill["floor_required"])
        self.assertIn("drill", drill["proxy_version"])


class TestLearningLedger(ArtifactTestCase):
    def test_exposures_link_loss_to_a_source_shard(self):
        evs = self.main.learning.events("shard_exposure")
        self.assertTrue(evs)
        for e in evs[:50]:
            self.assertTrue(e["shard_id"])
            self.assertIn(e["lane"], plan.LANES)
            self.assertIsNotNone(e["mean_token_loss"])

    def test_probe_deltas_were_actually_measured(self):
        evs = [e for e in self.main.learning.events("shard_exposure")
               if e.get("probe_loss_delta") is not None]
        self.assertTrue(evs)
        for e in evs[:50]:
            self.assertAlmostEqual(
                e["probe_loss_before"] - e["probe_loss_after"],
                e["probe_loss_delta"], places=5)
        self.assertGreater(len({e["probe_loss_delta"] for e in evs}), 5)

    def test_token_trace_records_carry_per_token_perplexity(self):
        traces = self.main.token_trace.events("token_trace")
        self.assertTrue(traces)
        import math
        for t in traces:
            for r in t["records"]:
                self.assertTrue(r["shard_id"])
                self.assertTrue(r["document_id"])
                self.assertEqual(r["loss_mask"], 1)
                self.assertAlmostEqual(math.exp(min(20.0, r["cross_entropy"])),
                                       r["perplexity"], places=2)

    def test_repeated_pass_numbers_are_tracked(self):
        passes = {e["repeated_pass"] for e in self.main.learning.events("shard_exposure")}
        self.assertTrue(passes)
        self.assertGreaterEqual(max(passes), 1)


class TestPerformance(ArtifactTestCase):
    def test_every_rate_is_reconstructible_from_the_raw_counts(self):
        wall = self.perf["wall_clock_s"]
        counts = self.perf["counts"]
        thr = self.perf["throughput"]
        for rate, count in [("raw_tokens_per_s", "token_positions"),
                            ("real_tokens_per_s", "real_tokens"),
                            ("useful_loss_bearing_tokens_per_s", "loss_bearing_tokens")]:
            self.assertAlmostEqual(counts[count] / wall, thr[rate], places=1, msg=rate)

    def test_token_accounting_is_internally_consistent(self):
        c = self.perf["counts"]
        self.assertLessEqual(c["loss_bearing_tokens"], c["real_tokens"])
        self.assertLessEqual(c["real_tokens"], c["token_positions"])
        self.assertEqual(c["padding_positions"],
                         c["token_positions"] - c["real_tokens"])
        self.assertEqual(c["context_only_tokens"],
                         c["real_tokens"] - c["loss_bearing_tokens"])

    def test_reported_utilization_agrees_with_the_ledger(self):
        real = sum(e.get("n_real_tokens", 0)
                   for e in self.main.consumption.events("optimizer_step"))
        pos = sum(e.get("n_positions", 0)
                  for e in self.main.consumption.events("optimizer_step"))
        self.assertAlmostEqual(real / pos,
                               self.perf["packing_utilization_from_ledger"], places=5)

    def test_concatenating_beats_padding_on_utilization(self):
        cmp = self.perf["packing_policy_comparison"]
        pairs = [(k, v) for k, v in cmp.items() if k.endswith("/concat_and_chop")]
        self.assertTrue(pairs)
        for key, concat in pairs:
            lane = key.split("/")[0]
            pad = cmp[f"{lane}/pad_only"]
            self.assertGreaterEqual(concat["mean_utilization"],
                                    pad["mean_utilization"], lane)


class TestEvidenceBundle(ArtifactTestCase):
    def test_all_requirements_pass(self):
        s = self.evidence["summary"]
        self.assertEqual(s["failed"], 0,
                         [r["key"] for r in self.evidence["requirements"]
                          if r["result"] == "FAIL"])
        self.assertTrue(s["all_passed"])

    def test_every_requirement_names_its_evidence_files(self):
        for r in self.evidence["requirements"]:
            self.assertTrue(r["evidence"], r["key"])
            self.assertTrue(r["how_checked"], r["key"])

    def test_evidence_agrees_with_an_independent_reading_of_the_ledger(self):
        """The bundle and these tests must reach the same conclusions."""
        by_key = {r["key"]: r for r in self.evidence["requirements"]}

        rep = AU.replay_interval(self.cat, self.main.consumption,
                                 *self.profile["replay_interval"],
                                 self.profile["seq_len"])
        self.assertEqual(by_key["replay"]["detail"]["n_microbatches_replayed"],
                         rep["n_microbatches_replayed"])

        floor = AU.floor_compliance(self.main.consumption)
        self.assertEqual(by_key["mixture_compliance"]["detail"]["floor_violations"],
                         floor["n_violations"])

        trail = AU.opus_audit(self.main.opus)
        self.assertEqual(by_key["opus_audit_trail"]["detail"]["n_decisions"],
                         trail["n_decisions"])

    def test_evidence_markdown_lists_every_requirement(self):
        md = (ART / "evidence.md").read_text(encoding="utf-8")
        for r in self.evidence["requirements"]:
            self.assertIn(r["requirement"], md, r["key"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
