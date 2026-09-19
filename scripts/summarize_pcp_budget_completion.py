"""Export a complete frozen D4 report and compute accounting without policy rollouts.

This reporting layer preserves the analyzer's intervals and decisions. It does
not issue an approval, reinterpret incomplete evidence, or recompute statistics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path


MODELS = ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0", "pcp_broadcast_k3")
VISIBILITIES = ("radius1", "global")
SEEDS = list(range(20, 25))
PRIMARY = "broadcast_k3_minus_pcp_local_capacity/radius1"
SOURCE = "69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e"
PROTOCOL = "ec7c109413625f270b79bffc1753adf5b7691d008eb90a4297d844b020898b0a"
PLAN = "5b1474bca7f4277efc71fa3ec7f2359c915a6cf5ebf71e1e1d356db51b7db4ba"
EXPECTED_OUTPUTS = {
    "analysis_report.json", "per_seed.csv", "episode_metrics.csv", "training_curves.csv",
    *(f"{name}.{extension}" for name in (
        "training_returns", "heldout_outcomes", "tested_budget_frontier")
      for extension in ("pdf", "svg", "png")),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), f"Expected an ordinary file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path, hashes, expected=None):
    path = Path(path)
    digest = sha(path)
    path = path.resolve()
    need(expected is None or digest == expected, f"Bound input changed: {path}")
    need(str(path) not in hashes or hashes[str(path)] == digest,
         f"Previously bound input changed: {path}")
    hashes[str(path)] = digest
    return path


def read(path, hashes):
    path = bind(path, hashes)
    return json.loads(path.read_text())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def interval(record):
    need(isinstance(record, dict), "A required five-seed interval is unavailable.")
    for key in ("mean", "ci95_low", "ci95_high"):
        need(number(record.get(key)), f"Missing/nonfinite interval {key}.")
    need(record["ci95_low"] <= record["ci95_high"], "Reversed interval endpoints.")
    need(record.get("training_seeds") == 5 and record.get("bootstrap_samples") == 10000
         and record.get("bootstrap_seed") == 73191 and record.get("unit") == "training_seed"
         and record.get("method") == "paired_seed_percentile_bootstrap",
         "The reported interval differs from the frozen uncertainty rule.")
    return record


def expected_contrasts():
    keys = {(f"broadcast_k3_minus_{model}/{visibility}", "contact_by_deadline")
            for visibility in VISIBILITIES
            for model in ("pcp_local_capacity", "pcp_broadcast_k0", "pcp_comm_identity")}
    keys.update((f"broadcast_k3_live_minus_{arm}/{visibility}", metric)
                for visibility in VISIBILITIES
                for arm in ("severed", "suppress_0", "suppress_1", "suppress_2", "shuffle")
                for metric in ("contact_by_deadline", "return"))
    keys.add(("capacity_controlled_visibility_interaction_radius1_minus_global",
              "contact_by_deadline"))
    return keys


def validate_report(report):
    need(report.get("schema_version") == 1
         and report.get("analysis") == "pcp_visibility_budget_seed_analysis_v1"
         and report.get("checks_passed") is True
         and report.get("status") == "complete_for_review"
         and report.get("issues") == [] and report.get("expected_policies") == 40
         and report.get("valid_policies") == 40,
         "A complete, valid forty-policy analyzer report is required.")
    need(report.get("approval_decision") == "not_issued", "Unexpected analyzer approval claim.")
    need(report.get("training_seeds") == SEEDS
         and report.get("heldout_episode_seeds") == list(range(200000, 200256))
         and report.get("deadline") == 50 and report.get("target") == 0.8
         and report.get("protocol_sha256") == PROTOCOL and report.get("plan_sha256") == PLAN
         and report.get("training_source_sha256") == SOURCE
         and report.get("analysis_source_sha256") == SOURCE
         and report.get("primary_contrast") == PRIMARY,
         "The report is not the frozen D4 allocation.")
    rows = report["rows"]
    expected = set(itertools.product(MODELS, VISIBILITIES, SEEDS))
    need(len(rows) == 40 and {(r["model"], r["visibility"], r["seed"]) for r in rows}
         == expected and len({r["run_id"] for r in rows}) == 40,
         "Missing or duplicated policy identity.")
    for row in rows:
        need(row["status"] == "valid" and row["validation_passed"] is True
             and row["issues"] == [], "An individual policy is invalid.")
    contrasts = {(r["contrast"], r["metric"]): r for r in report["contrasts"]}
    need(len(report["contrasts"]) == 27 and set(contrasts) == expected_contrasts(),
         "Missing, duplicated or unexpected contrast/metric identity.")
    for row in contrasts.values():
        need(row["status"] == "complete" and row["missing_or_invalid_seeds"] == []
             and [p["seed"] for p in row["per_seed"]] == SEEDS
             and all(number(p["value"]) for p in row["per_seed"]),
             "A contrast lacks all five finite paired seed effects.")
        interval(row["interval"])
    conditions = {(r["model"], r["visibility"]): r for r in report["conditions"]}
    need(len(report["conditions"]) == 8
         and set(conditions) == set(itertools.product(MODELS, VISIBILITIES)),
         "Missing or duplicated method/visibility condition.")
    for row in conditions.values():
        need(row["status"] == "complete" and row["valid_seeds"] == SEEDS,
             "Incomplete five-seed condition.")
        interval(row["intervals"]["contact_by_deadline"])
    frontiers = {r["visibility"]: r for r in report["budget_frontiers"]}
    need(len(report["budget_frontiers"]) == 2 and set(frontiers) == set(VISIBILITIES),
         "Missing or duplicated budget frontier.")
    for row in frontiers.values():
        need(row["target"] == 0.8 and row["deadline"] == 50
             and [b["sender_budget"] for b in row["budgets"]] == [0, 3],
             "Unexpected tested budget rule.")
        for budget in row["budgets"]:
            lower = interval(budget["interval"])["ci95_low"]
            need(budget["qualifies"] is (lower >= 0.8),
                 "Published budget qualification disagrees with its lower bound.")
        qualified = [b["sender_budget"] for b in row["budgets"] if b["qualifies"]]
        expected = min(qualified) if qualified else "not reached"
        need(type(row["smallest_tested_budget"]) is type(expected)
             and row["smallest_tested_budget"] == expected,
             "Published smallest tested budget differs from the frozen rule.")
    return contrasts, conditions, frontiers


def elapsed(start, finish):
    a, b = datetime.fromisoformat(start), datetime.fromisoformat(finish)
    need(a.tzinfo is not None and b.tzinfo is not None, "Elapsed-time input needs time zones.")
    seconds = (b - a).total_seconds()
    need(seconds >= 0, "Negative elapsed time.")
    return seconds


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def generate(analysis_dir, queue_path, destination):
    analysis_dir, queue_path, destination = (
        Path(p).resolve() for p in (analysis_dir, queue_path, destination))
    need(not destination.exists(), "Preserve an existing reporting attempt; no overwrite.")
    need(not destination.is_relative_to(analysis_dir)
         and not analysis_dir.is_relative_to(destination)
         and not queue_path.is_relative_to(destination), "Output overlaps protected inputs.")
    hashes = {}
    script_path = bind(Path(__file__), hashes)
    report = read(analysis_dir / "analysis_report.json", hashes)
    contrasts, conditions, frontiers = validate_report(report)
    provenance = read(analysis_dir / "provenance.json", hashes)
    need(set(provenance["output_artifact_hashes"]) == EXPECTED_OUTPUTS
         and provenance["input_artifact_hashes"] == report["artifact_hashes"]
         and provenance["no_training_or_policy_rollout"] is True
         and provenance.get("schema_version") == 1
         and provenance.get("training_source_sha256") == SOURCE
         and provenance.get("analysis_source_sha256") == SOURCE,
         "Analyzer provenance or exact output inventory differs.")
    analyzers = [path for path in report["artifact_hashes"]
                 if Path(path).name == "analyze_pcp_budget.py"]
    need(len(analyzers) == 1 and provenance.get("script_sha256")
         == report["artifact_hashes"][analyzers[0]], "Analyzer script provenance differs.")
    for name, digest in provenance["output_artifact_hashes"].items():
        path = analysis_dir / name
        need(sha(path) == digest, f"Analyzer output changed: {name}")
        bind(path, hashes, digest)
    for name, digest in report["artifact_hashes"].items():
        need(sha(Path(name)) == digest, f"Frozen analyzer input changed: {name}")
        bind(Path(name), hashes, digest)
    queue = read(queue_path, hashes)
    need(queue["status"] == "completed" and queue["phase"] == "final_analysis"
         and set(queue["rows"]) == {r["run_id"] for r in report["rows"]},
         "The entire queue has not closed through final analysis.")
    manifest = Path(queue["manifest"])
    need(report["artifact_hashes"].get(str(manifest.resolve())) == queue["manifest_sha256"],
         "Queue manifest is not bound by this analyzer report.")
    manifest = bind(manifest, hashes, queue["manifest_sha256"])
    compute, training = [], []
    for row in report["rows"]:
        q = queue["rows"][row["run_id"]]
        need(q["status"] == "completed" and q["stage"] == "complete"
             and q["training_completed"] is True and q["evaluation_checks_passed"] is True
             and q["archive"]["checks_passed"] is True, "A row lacks final preservation gates.")
        archive = q["archive"]
        receipt_path = Path(archive["destination"]) / "archive_manifest.json"
        receipt = read(receipt_path, hashes)
        archive_path = Path(archive["destination"]) / "run.tar.gz"
        expected_receipt = {"schema_version": 1, "checks_passed": True,
                            "run_id": row["run_id"], "archive_file": "run.tar.gz",
                            "source_before_after_equal": True,
                            "all_archive_files_rehashed": True, "original_files_removed": False}
        need(all(type(receipt.get(key)) is type(value) and receipt[key] == value
                 for key, value in expected_receipt.items()),
             "Archive receipt identity or preservation gate differs.")
        need(not (Path(archive["destination"]) / "archive_failure.json").exists(),
             "Archive contains a failure marker.")
        need(sha(receipt_path) == archive["receipt_sha256"]
             and receipt["archive_sha256"] == archive["archive_sha256"] == sha(archive_path)
             and receipt["archive_bytes"] == archive["archive_bytes"] == archive_path.stat().st_size
             and receipt["checks_passed"] is True, "Archive receipt or bytes changed.")
        bind(archive_path, hashes, archive["archive_sha256"])
        reused = (row["model"] in ("pcp_comm_identity", "pcp_local_capacity")
                  and row["visibility"] == "radius1" and row["seed"] == 20)
        need(q["training_reused"] is reused,
             "Training reuse does not match the two historical policies.")
        if q["training_reused"]:
            reuse_path = Path(archive["destination"]) / "archive_reuse_receipt.json"
            reuse = read(reuse_path, hashes)
            need(sha(reuse_path) == archive["reuse_receipt_sha256"]
                 and archive.get("reused_original_archive") is True
                 and reuse["checks_passed"] is True and reuse["schema_version"] == 1
                 and reuse["run_id"] == row["run_id"]
                 and Path(reuse["destination"]).resolve() == receipt_path.parent.resolve()
                 and reuse["historical_files_modified"] is False
                 and reuse["original_files_removed"] is False
                 and reuse["all_copied_files_rehashed"] is True
                 and reuse["copied_archive_readback_verified"] is True,
                 "Historical archive reuse receipt changed.")
            source = reuse["source_verification"]
            need(source["checks_passed"] is True and source["run_id"] == row["run_id"]
                 and source["archive_sha256"] == archive["archive_sha256"]
                 and source["receipt_sha256"] == archive["receipt_sha256"]
                 and source["archive_bytes"] == archive["archive_bytes"]
                 and source["original_approval_matches_metadata"] is True
                 and source["all_archive_files_rehashed"] is True
                 and source["source_before_after_equal"] is True,
                 "Historical archive source verification differs.")
            binding = reuse["operational_binding_sha256"]
            need(binding.get(str(manifest)) == queue["manifest_sha256"],
                 "Historical reuse does not bind the analyzed manifest.")
            for artifact_map in (source["input_artifact_sha256"], binding):
                for name, digest in artifact_map.items():
                    bind(Path(name), hashes, digest)
        else:
            need(archive.get("reused_original_archive", False) is False,
                 "New training unexpectedly declares a reused archive.")
        metadata_path = Path(row["run_directory"]) / "metadata.json"
        need(str(metadata_path.resolve()) in report["artifact_hashes"],
             "Training metadata is not an analyzer-bound input.")
        metadata = read(metadata_path, hashes)
        need(metadata["status"] == "completed" and metadata["frames"] == 600000
             and metadata["iterations"] == 100 and metadata["run_id"] == row["run_id"]
             and metadata["model"] == row["model"] and metadata["seed"] == row["seed"]
             and metadata["ablation_value"] == row["visibility"]
             and metadata["parameters"] == row["parameters"],
             "Unexpected final training metadata.")
        seconds = metadata["benchmarl_total_time_seconds"]
        need(number(seconds) and seconds >= 0, "Missing recorded BenchMARL elapsed time.")
        identity = {k: row[k] for k in ("run_id", "model", "visibility", "seed")}
        compute.append({
            **identity, "frames": metadata["frames"], "iterations": metadata["iterations"],
            "benchmarl_elapsed_seconds": seconds,
            "managed_training_elapsed_seconds": elapsed(
                metadata["start_timestamp"], metadata["end_timestamp"]),
            "continuation_row_elapsed_seconds": elapsed(q["started_utc"], q["finished_utc"]),
            "training_reused": q["training_reused"],
            "original_training_hostname": metadata["runtime"]["hostname"],
            "archive_bytes": archive["archive_bytes"],
            "parameters_json": json.dumps(row["parameters"], sort_keys=True),
        })
        performance = row["performance"]
        need(performance["final_window_frames"] == list(range(540000, 600001, 12000))
             and performance["auc_frame_interval"] == [6000, 600000]
             and performance["first_logged_evaluation_frames"] == 6000
             and all(number(performance[key]) for key in (
                 "final_window_mean_return", "normalized_auc", "first_logged_evaluation_return",
                 "final_evaluation_return")),
             "Training performance differs from the frozen horizon.")
        training.append({
            **identity, "final_six_mean_return": performance["final_window_mean_return"],
            "final_six_frames_json": json.dumps(performance["final_window_frames"]),
            "normalized_auc": performance["normalized_auc"],
            "auc_frame_interval_json": json.dumps(performance["auc_frame_interval"]),
            "first_logged_evaluation_return": performance["first_logged_evaluation_return"],
            "first_logged_evaluation_frames": performance["first_logged_evaluation_frames"],
            "final_evaluation_return": performance["final_evaluation_return"],
        })
    exported_contrasts = [{
        "contrast": name, "metric": metric, **row["interval"],
        "paired_seed_effects_json": json.dumps(row["per_seed"], sort_keys=True),
        "terms_json": json.dumps(row["terms"], sort_keys=True),
    } for (name, metric), row in sorted(contrasts.items())]
    primary = contrasts[(PRIMARY, "contact_by_deadline")]
    summary = {
        "schema_version": 1, "status": "complete_evidence_reported",
        "analyzer_approval_decision": report["approval_decision"],
        "primary_contrast": primary,
        "smallest_tested_budgets": {key: row["smallest_tested_budget"]
                                   for key, row in frontiers.items()},
        "training_frames": sum(r["frames"] for r in compute),
        "sum_benchmarl_elapsed_seconds": sum(r["benchmarl_elapsed_seconds"] for r in compute),
        "continuation_elapsed_seconds": elapsed(queue["started_utc"], queue["finished_utc"]),
        "compute_scope": "Per-run BenchMARL elapsed time includes scheduled online evaluation. "
                         "Its sum is not cohort wall time or measured CPU time. Continuation row "
                         "times include recovery or training/evaluation/archive service; the two "
                         "original trainings and recovered banks occurred before this "
                         "continuation. "
                         "Separate held-out evaluator CPU/elapsed times are not reported per row.",
        "archive_scope": "All forty archive hashes/receipts rechecked; member readback is retained "
                         "in the original per-run gates and requires separate independent review.",
        "no_statistics_recomputed": True, "no_training_or_policy_rollout": True,
    }
    ci = primary["interval"]
    lines = [
        "# D4 complete-cohort reporting tables", "",
        "All 40 allocated policies have complete analyzer and queue evidence. "
        "The analyzer leaves approval `not_issued`; evidence completeness is not "
        "a positive effect.",
        "", f"The primary restricted-visibility K3 minus LocalCapacity difference in contact "
        f"by step 50 is **{100 * ci['mean']:+.3f} percentage points** "
        f"(descriptive 95% paired-seed bootstrap interval "
        f"{100 * ci['ci95_low']:+.3f} to {100 * ci['ci95_high']:+.3f}).", "",
        "| Visibility | Method | Contact by 50 | 95% interval | Mean return |",
        "|---|---|---:|---:|---:|",
    ]
    for visibility, model in itertools.product(VISIBILITIES, MODELS):
        condition = conditions[(model, visibility)]
        value = condition["intervals"]["contact_by_deadline"]
        lines.append(f"| {visibility} | {model} | {value['mean']:.6f} | "
                     f"{value['ci95_low']:.6f}–{value['ci95_high']:.6f} | "
                     f"{condition['means']['return']:.6f} |")
    lines += ["", "| Visibility | Smallest qualifying tested budget |", "|---|---|",
              *(f"| {v} | {frontiers[v]['smallest_tested_budget']} |" for v in VISIBILITIES),
              "", "Qualification requires the five-seed 95% lower bound to reach 0.80. "
              "Only K0 and K3 were tested; K1/K2 remain untested.", "",
              "Each policy used 600,000 frames; the complete allocation is 24,000,000 frames. "
              "The independent units are five paired training seeds, with episodes averaged "
              "within each policy. The reported intervals and 27 contrasts are copied from "
              "the frozen analyzer without statistical recomputation.", "",
              "Severing, suppression and cyclic donor substitution measure reliance of the "
              "same final policy under intervention and associated distribution shift. They "
              "do not establish task impossibility without communication. Contact rewards "
              "count repeated contacts; global prey visibility is not a full-state actor. "
              "No convergence, untested bandwidth bound or population superiority follows.", "",
              summary["compute_scope"], "", summary["archive_scope"], ""]
    need(all(sha(Path(path)) == digest for path, digest in hashes.items()),
         "A reporting input changed before publication.")
    destination.mkdir(parents=True)
    write_csv(destination / "contrasts.csv", exported_contrasts)
    write_csv(destination / "training_performance.csv", training)
    write_csv(destination / "compute_accounting.csv", compute)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True,
                                                         allow_nan=False) + "\n")
    (destination / "RESULTS.md").write_text("\n".join(lines))
    need(all(sha(Path(path)) == digest for path, digest in hashes.items()),
         "A reporting input changed during publication; preserve this attempt.")
    evidence = {"checks_passed": True, "input_artifact_sha256": hashes,
                "output_artifact_sha256": {p.name: sha(p) for p in sorted(destination.iterdir())},
                "script_sha256": hashes[str(script_path)], "no_training_or_policy_rollout": True,
                "no_statistics_recomputed": True}
    (destination / "provenance.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--queue-status", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate(args.analysis_dir, args.queue_status, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
