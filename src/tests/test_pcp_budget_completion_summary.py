"""Reporting guards use synthetic evidence only; no scientific policy is evaluated."""

import copy
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "completion_summary", Path(__file__).resolve().parents[2]
    / "scripts/summarize_pcp_budget_completion.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def interval(mean=0.9, low=0.85, high=0.95):
    return {"mean": mean, "ci95_low": low, "ci95_high": high,
            "training_seeds": 5, "bootstrap_samples": 10000, "bootstrap_seed": 73191,
            "unit": "training_seed", "method": "paired_seed_percentile_bootstrap"}


def complete_report():
    return {
        "schema_version": 1, "analysis": "pcp_visibility_budget_seed_analysis_v1",
        "checks_passed": True, "status": "complete_for_review", "issues": [],
        "expected_policies": 40, "valid_policies": 40, "approval_decision": "not_issued",
        "training_seeds": summary.SEEDS, "heldout_episode_seeds": list(range(200000, 200256)),
        "deadline": 50, "target": 0.8, "protocol_sha256": summary.PROTOCOL,
        "plan_sha256": summary.PLAN, "training_source_sha256": summary.SOURCE,
        "analysis_source_sha256": summary.SOURCE, "primary_contrast": summary.PRIMARY,
        "artifact_hashes": {},
        "rows": [{"model": m, "visibility": v, "seed": seed,
                  "run_id": f"{m}_{v}_{seed}", "status": "valid",
                  "validation_passed": True, "issues": []}
                 for m, v, seed in itertools.product(summary.MODELS, summary.VISIBILITIES,
                                                      summary.SEEDS)],
        "contrasts": [{"contrast": name, "metric": metric, "status": "complete",
                       "missing_or_invalid_seeds": [], "interval": interval(),
                       "terms": [{"coefficient": 1, "model": "synthetic_term"}],
                       "per_seed": [{"seed": seed, "value": 0.9} for seed in summary.SEEDS]}
                      for name, metric in sorted(summary.expected_contrasts())],
        "conditions": [{"model": m, "visibility": v, "status": "complete",
                        "valid_seeds": summary.SEEDS,
                        "means": {"return": 100.0},
                        "intervals": {"contact_by_deadline": interval()}}
                       for m, v in itertools.product(summary.MODELS, summary.VISIBILITIES)],
        "budget_frontiers": [{"visibility": v, "target": 0.8, "deadline": 50,
                              "budgets": [{"sender_budget": k, "interval": interval(),
                                           "qualifies": True} for k in (0, 3)],
                              "smallest_tested_budget": 0} for v in summary.VISIBILITIES],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def complete_packet(root):
    """Forty synthetic metadata/receipt records; no policy, archive extraction or rollout."""
    root = root.resolve()
    analysis, inputs, runs, archives = (root / name for name in
                                       ("analysis", "inputs", "runs", "archives"))
    for directory in (analysis, inputs, runs, archives):
        directory.mkdir()
    manifest = inputs / "manifest.jsonl"
    manifest.write_text("synthetic frozen manifest fixture\n")
    analyzer = inputs / "analyze_pcp_budget.py"
    analyzer.write_text("# synthetic provenance fixture; never executed\n")
    relocation = inputs / "relocation_receipt.json"
    write_json(relocation, {"synthetic": True})
    report = complete_report()
    report["artifact_hashes"] = {str(path): summary.sha(path) for path in (manifest, analyzer)}
    queue = {"status": "completed", "phase": "final_analysis", "manifest": str(manifest),
             "manifest_sha256": summary.sha(manifest), "rows": {},
             "started_utc": "2026-09-10T01:00:00+00:00",
             "finished_utc": "2026-09-10T02:00:00+00:00"}
    for index, row in enumerate(report["rows"]):
        run = runs / row["run_id"]
        archive_dir = archives / row["run_id"]
        run.mkdir()
        archive_dir.mkdir()
        row["run_directory"] = str(run)
        row["parameters"] = {"actor": 100, "communication_branch": 0, "critic": 200,
                             "total": 300}
        row["performance"] = {
            "final_window_mean_return": 10.0 + index,
            "final_window_frames": list(range(540000, 600001, 12000)),
            "normalized_auc": 5.0 + index, "auc_frame_interval": [6000, 600000],
            "first_logged_evaluation_return": 1.0 + index,
            "first_logged_evaluation_frames": 6000, "final_evaluation_return": 12.0 + index}
        reused = (row["model"] in ("pcp_comm_identity", "pcp_local_capacity")
                  and row["visibility"] == "radius1" and row["seed"] == 20)
        day = "08" if reused else "10"
        metadata = {"run_id": row["run_id"], "model": row["model"], "seed": row["seed"],
                    "ablation_value": row["visibility"], "status": "completed",
                    "frames": 600000, "iterations": 100, "parameters": row["parameters"],
                    "benchmarl_total_time_seconds": 10.0,
                    "start_timestamp": f"2026-09-{day}T01:00:00+00:00",
                    "end_timestamp": f"2026-09-{day}T01:00:12+00:00",
                    "runtime": {"hostname": "original-host" if reused else "current-host"}}
        metadata_path = run / "metadata.json"
        write_json(metadata_path, metadata)
        report["artifact_hashes"][str(metadata_path)] = summary.sha(metadata_path)
        archive_path = archive_dir / "run.tar.gz"
        archive_path.write_bytes(f"synthetic archive bytes for {row['run_id']}".encode())
        receipt = {"schema_version": 1, "checks_passed": True, "run_id": row["run_id"],
                   "archive_file": "run.tar.gz", "archive_sha256": summary.sha(archive_path),
                   "archive_bytes": archive_path.stat().st_size,
                   "source_before_after_equal": True, "all_archive_files_rehashed": True,
                   "original_files_removed": False}
        receipt_path = archive_dir / "archive_manifest.json"
        write_json(receipt_path, receipt)
        archive = {"destination": str(archive_dir), "checks_passed": True,
                   "receipt_sha256": summary.sha(receipt_path),
                   "archive_sha256": summary.sha(archive_path),
                   "archive_bytes": archive_path.stat().st_size}
        if reused:
            source = {"checks_passed": True, "run_id": row["run_id"],
                      "archive_sha256": archive["archive_sha256"],
                      "receipt_sha256": archive["receipt_sha256"],
                      "archive_bytes": archive["archive_bytes"],
                      "original_approval_matches_metadata": True,
                      "all_archive_files_rehashed": True, "source_before_after_equal": True,
                      "input_artifact_sha256": {str(metadata_path): summary.sha(metadata_path)}}
            reuse = {"schema_version": 1, "checks_passed": True, "run_id": row["run_id"],
                     "destination": str(archive_dir), "historical_files_modified": False,
                     "original_files_removed": False, "all_copied_files_rehashed": True,
                     "copied_archive_readback_verified": True, "source_verification": source,
                     "operational_binding_sha256": {str(path): summary.sha(path)
                                                    for path in (manifest, relocation)}}
            reuse_path = archive_dir / "archive_reuse_receipt.json"
            write_json(reuse_path, reuse)
            archive.update(reused_original_archive=True,
                           reuse_receipt_sha256=summary.sha(reuse_path))
        queue["rows"][row["run_id"]] = {
            "status": "completed", "stage": "complete", "training_completed": True,
            "evaluation_checks_passed": True, "training_reused": reused, "archive": archive,
            "started_utc": "2026-09-10T01:00:00+00:00",
            "finished_utc": "2026-09-10T01:00:30+00:00"}
    # The duplicate intervention name has deliberately different units and values.
    for contrast in report["contrasts"]:
        if contrast["metric"] == "return":
            contrast["interval"] = interval(mean=12.0, low=10.0, high=14.0)
            contrast["per_seed"] = [{"seed": seed, "value": 12.0} for seed in summary.SEEDS]
    for name in summary.EXPECTED_OUTPUTS:
        (analysis / name).write_bytes(b"synthetic analyzer output fixture")
    write_json(analysis / "analysis_report.json", report)
    provenance = {"schema_version": 1, "training_source_sha256": summary.SOURCE,
                  "analysis_source_sha256": summary.SOURCE,
                  "script_sha256": summary.sha(analyzer),
                  "input_artifact_hashes": report["artifact_hashes"],
                  "output_artifact_hashes": {name: summary.sha(analysis / name)
                                             for name in summary.EXPECTED_OUTPUTS},
                  "no_training_or_policy_rollout": True}
    write_json(analysis / "provenance.json", provenance)
    queue_path = root / "queue.json"
    write_json(queue_path, queue)
    return analysis, queue_path, report, queue


class CompletionSummaryTests(unittest.TestCase):
    def test_zero_budget_is_preserved_and_both_intervention_metrics_remain_distinct(self):
        contrasts, _, frontiers = summary.validate_report(complete_report())
        self.assertEqual(frontiers["radius1"]["smallest_tested_budget"], 0)
        name = "broadcast_k3_live_minus_severed/radius1"
        self.assertIn((name, "contact_by_deadline"), contrasts)
        self.assertIn((name, "return"), contrasts)
        self.assertEqual(len(contrasts), 27)

    def test_incomplete_or_duplicate_policy_cannot_be_reported_as_complete(self):
        for change in (lambda r: r.update(valid_policies=39),
                       lambda r: r["rows"].pop(),
                       lambda r: r["rows"].__setitem__(-1, copy.deepcopy(r["rows"][0])),
                       lambda r: r["rows"][0].update(issues=["missing bank"]),
                       lambda r: r.update(checks_passed=False)):
            report = complete_report()
            change(report)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.validate_report(report)

    def test_duplicate_metric_cannot_hide_a_missing_contrast(self):
        report = complete_report()
        candidates = [r for r in report["contrasts"]
                      if r["contrast"] == "broadcast_k3_live_minus_severed/radius1"]
        candidates[1]["metric"] = candidates[0]["metric"]
        with self.assertRaisesRegex(ValueError, "contrast/metric identity"):
            summary.validate_report(report)

    def test_reduced_seed_interval_and_nonfinite_effect_are_rejected(self):
        for change in (lambda r: r["contrasts"][0]["interval"].update(training_seeds=4),
                       lambda r: r["contrasts"][0]["per_seed"].pop(),
                       lambda r: r["contrasts"][0]["per_seed"][0].update(value=float("nan")),
                       lambda r: r["contrasts"][0].update(missing_or_invalid_seeds=[24])):
            report = complete_report()
            change(report)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.validate_report(report)

    def test_point_estimate_above_target_cannot_qualify_a_low_lower_bound(self):
        report = complete_report()
        for frontier in report["budget_frontiers"]:
            for budget in frontier["budgets"]:
                budget.update(interval=interval(mean=0.85, low=0.79, high=0.92), qualifies=False)
            frontier["smallest_tested_budget"] = "not reached"
        _, _, frontiers = summary.validate_report(report)
        self.assertEqual(frontiers["global"]["smallest_tested_budget"], "not reached")
        report["budget_frontiers"][0]["budgets"][0]["qualifies"] = True
        with self.assertRaisesRegex(ValueError, "qualification disagrees"):
            summary.validate_report(report)

    def test_relabelled_approval_or_changed_frozen_rule_is_rejected(self):
        for change in ({"approval_decision": "approved"}, {"deadline": 49}, {"target": 0.75},
                      {"analysis_source_sha256": "different"}, {"training_seeds": [20, 21]}):
            report = complete_report()
            report.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.validate_report(report)

    def test_missing_cohort_and_changed_analyzer_output_publish_no_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = root / "analysis"
            analysis.mkdir()
            destination = root / "report"
            report = complete_report()
            report["valid_policies"] = 39
            (analysis / "analysis_report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "complete, valid"):
                summary.generate(analysis, root / "queue.json", destination)
            self.assertFalse(destination.exists())
            report["valid_policies"] = 40
            for name in summary.EXPECTED_OUTPUTS:
                (analysis / name).write_bytes(b"synthetic output only")
            (analysis / "analysis_report.json").write_text(json.dumps(report))
            analyzer = root / "analyze_pcp_budget.py"
            analyzer.write_text("synthetic analyzer")
            report["artifact_hashes"] = {str(analyzer): summary.sha(analyzer)}
            (analysis / "analysis_report.json").write_text(json.dumps(report))
            provenance = {"schema_version": 1, "training_source_sha256": summary.SOURCE,
                          "analysis_source_sha256": summary.SOURCE,
                          "script_sha256": summary.sha(analyzer),
                          "output_artifact_hashes": {
                name: summary.sha(analysis / name) for name in summary.EXPECTED_OUTPUTS},
                "input_artifact_hashes": report["artifact_hashes"],
                "no_training_or_policy_rollout": True}
            (analysis / "provenance.json").write_text(json.dumps(provenance))
            (analysis / "per_seed.csv").write_bytes(b"changed after analyzer")
            with self.assertRaisesRegex(ValueError, "Analyzer output changed"):
                summary.generate(analysis, root / "queue.json", destination)
            self.assertFalse(destination.exists())

    def test_complete_generate_preserves_units_zero_budget_time_scope_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis, queue_path, report, queue = complete_packet(root)
            destination = root / "published"
            result = summary.generate(analysis, queue_path, destination)
            self.assertEqual(result["training_frames"], 24000000)
            self.assertEqual(result["sum_benchmarl_elapsed_seconds"], 400)
            self.assertEqual(result["continuation_elapsed_seconds"], 3600)
            self.assertEqual(result["smallest_tested_budgets"], {"radius1": 0, "global": 0})
            self.assertEqual(result["primary_contrast"], next(
                row for row in report["contrasts"] if row["contrast"] == summary.PRIMARY))
            self.assertIn("not cohort wall time or measured CPU time", result["compute_scope"])
            self.assertEqual(result["analyzer_approval_decision"], "not_issued")
            with (destination / "contrasts.csv").open() as stream:
                contrasts = {(r["contrast"], r["metric"]): r for r in csv.DictReader(stream)}
            self.assertEqual(len(contrasts), 27)
            name = "broadcast_k3_live_minus_severed/radius1"
            self.assertEqual(float(contrasts[(name, "contact_by_deadline")]["mean"]), 0.9)
            self.assertEqual(float(contrasts[(name, "return")]["mean"]), 12.0)
            with (destination / "compute_accounting.csv").open() as stream:
                compute = list(csv.DictReader(stream))
            self.assertEqual(len(compute), 40)
            self.assertEqual(sum(r["training_reused"] == "True" for r in compute), 2)
            self.assertTrue(all(float(r["managed_training_elapsed_seconds"]) == 12
                                and float(r["continuation_row_elapsed_seconds"]) == 30
                                for r in compute))
            with (destination / "training_performance.csv").open() as stream:
                training = list(csv.DictReader(stream))
            self.assertEqual(len(training), 40)
            self.assertEqual(json.loads(training[0]["final_six_frames_json"]),
                             list(range(540000, 600001, 12000)))
            self.assertEqual(float(training[0]["normalized_auc"]), 5.0)
            markdown = (destination / "RESULTS.md").read_text()
            self.assertIn("| radius1 | 0 |", markdown)
            self.assertIn("+90.000 percentage points", markdown)
            self.assertIn("approval `not_issued`", markdown)
            provenance = json.loads((destination / "provenance.json").read_text())
            expected = {"contrasts.csv", "compute_accounting.csv", "training_performance.csv",
                        "summary.json", "RESULTS.md"}
            self.assertEqual(set(provenance["output_artifact_sha256"]), expected)
            for name, digest in provenance["output_artifact_sha256"].items():
                self.assertEqual(summary.sha(destination / name), digest)
            for name, digest in provenance["input_artifact_sha256"].items():
                self.assertEqual(summary.sha(Path(name)), digest)
            self.assertEqual(provenance["input_artifact_sha256"][queue["manifest"]],
                             queue["manifest_sha256"])
            self.assertEqual(sum(Path(name).name == "archive_manifest.json"
                                 for name in provenance["input_artifact_sha256"]), 40)
            self.assertEqual(sum(Path(name).name == "archive_reuse_receipt.json"
                                 for name in provenance["input_artifact_sha256"]), 2)

    def test_incomplete_queue_or_unbound_manifest_publishes_nothing(self):
        for fault in ("incomplete", "row_incomplete", "other_manifest"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                analysis, queue_path, _, queue = complete_packet(root)
                if fault == "incomplete":
                    queue["status"] = "running"
                elif fault == "row_incomplete":
                    next(iter(queue["rows"].values()))["evaluation_checks_passed"] = False
                else:
                    other = root / "other_manifest.jsonl"
                    other.write_bytes(Path(queue["manifest"]).read_bytes())
                    queue["manifest"] = str(other)
                write_json(queue_path, queue)
                with self.assertRaises(ValueError):
                    summary.generate(analysis, queue_path, root / "published")
                self.assertFalse((root / "published").exists())

    def test_swapped_archive_receipts_or_failure_marker_publishes_nothing(self):
        for fault in ("swapped", "failure", "modified_bytes"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                analysis, queue_path, _, queue = complete_packet(root)
                rows = [r for r in queue["rows"].values() if not r["training_reused"]]
                if fault == "swapped":
                    rows[0]["archive"], rows[1]["archive"] = rows[1]["archive"], rows[0]["archive"]
                    write_json(queue_path, queue)
                else:
                    archive = Path(rows[0]["archive"]["destination"])
                    if fault == "failure":
                        write_json(archive / "archive_failure.json", {"failure": "synthetic"})
                    else:
                        (archive / "run.tar.gz").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    summary.generate(analysis, queue_path, root / "published")
                self.assertFalse((root / "published").exists())

    def test_rehashed_reuse_receipt_still_requires_identity_and_manifest_binding(self):
        for fault in ("identity", "manifest", "source", "training_reuse"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                analysis, queue_path, _, queue = complete_packet(root)
                row = next(r for r in queue["rows"].values() if r["training_reused"])
                receipt_path = Path(row["archive"]["destination"]) / "archive_reuse_receipt.json"
                reuse = json.loads(receipt_path.read_text())
                if fault == "identity":
                    reuse["run_id"] = "a_different_policy"
                elif fault == "manifest":
                    reuse["operational_binding_sha256"].pop(queue["manifest"])
                elif fault == "source":
                    reuse["source_verification"]["original_approval_matches_metadata"] = False
                else:
                    row["training_reused"] = False
                write_json(receipt_path, reuse)
                row["archive"]["reuse_receipt_sha256"] = summary.sha(receipt_path)
                write_json(queue_path, queue)
                with self.assertRaises(ValueError):
                    summary.generate(analysis, queue_path, root / "published")
                self.assertFalse((root / "published").exists())

    def test_metadata_drift_cannot_replace_analyzer_hash_during_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis, queue_path, report, _ = complete_packet(root)
            metadata_path = Path(report["rows"][0]["run_directory"]) / "metadata.json"
            original_read = summary.read

            def drifting_read(path, hashes):
                if Path(path).resolve() == queue_path:
                    metadata = json.loads(metadata_path.read_text())
                    metadata["benchmarl_total_time_seconds"] = 999.0
                    write_json(metadata_path, metadata)
                return original_read(path, hashes)

            with patch.object(summary, "read", side_effect=drifting_read):
                with self.assertRaisesRegex(ValueError, "Bound input changed|Previously bound"):
                    summary.generate(analysis, queue_path, root / "published")
            self.assertFalse((root / "published").exists())

    def test_input_drift_during_publication_leaves_attempt_without_success_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis, queue_path, _, queue = complete_packet(root)
            original_write = summary.write_csv

            def drifting_write(path, rows):
                original_write(path, rows)
                if path.name == "compute_accounting.csv":
                    Path(queue["manifest"]).write_text("manifest changed during publication")

            with patch.object(summary, "write_csv", side_effect=drifting_write):
                with self.assertRaisesRegex(ValueError, "changed during publication"):
                    summary.generate(analysis, queue_path, root / "published")
            self.assertTrue((root / "published").exists())
            self.assertFalse((root / "published" / "provenance.json").exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "existing"
            out.mkdir()
            marker = out / "partial_evidence"
            marker.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "no overwrite"):
                summary.generate(root / "analysis", root / "queue.json", out)
            self.assertEqual(marker.read_text(), "preserve")

    def test_elapsed_times_are_aware_and_never_negative(self):
        self.assertEqual(summary.elapsed("2026-09-09T21:00:00-04:00",
                                         "2026-09-10T01:01:00+00:00"), 60)
        with self.assertRaises(ValueError):
            summary.elapsed("2026-09-09T21:00:00", "2026-09-09T21:01:00")
        with self.assertRaises(ValueError):
            summary.elapsed("2026-09-10T01:01:00+00:00", "2026-09-10T01:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
