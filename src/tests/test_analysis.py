from __future__ import annotations

import csv
import json

import pytest

from commstudy.analysis import (
    aggregate_suite,
    analyze_suite,
    bootstrap_mean_ci,
    compute_run_result,
)
from commstudy.experiments.metrics import TidyMetricsWriter


def _run(
    suite_dir,
    model,
    seed,
    values,
    status="completed",
    *,
    ablation="main",
    ablation_value=None,
    comm_bits=32,
):
    value_suffix = f"-{ablation_value}" if ablation_value is not None else ""
    run_dir = suite_dir / f"{model}{value_suffix}-{seed}"
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_dir.name,
        "suite_id": suite_dir.name,
        "status": status,
        "task": "vmas_simple_spread",
        "algorithm": "mappo",
        "model": model,
        "seed": seed,
        "ablation": ablation,
        "ablation_value": ablation_value,
        "benchmarl_total_time_seconds": 10 + seed,
        "parameters": {
            "actor_total": 100,
            "communication_total": 10,
            "critic_total": 50,
        },
    }
    if status != "completed":
        metadata["error"] = {"type": "RuntimeError", "message": "failed"}
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    writer = TidyMetricsWriter(run_dir)
    for index, value in enumerate(values):
        writer.write(
            frames=(index + 1) * 6_000,
            iteration=index,
            phase="evaluation",
            metrics={"return_mean": value, "comm_message_bits": comm_bits},
        )
    return run_dir


def _manifest(suite_dir, rows):
    suite_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "suite_id",
        "task",
        "algorithm",
        "model",
        "seed",
        "ablation",
        "ablation_value",
        "status",
        "attempt",
    ]
    with (suite_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "suite_id": suite_dir.name,
                    "task": "vmas_simple_spread",
                    "algorithm": "mappo",
                    "model": row["model"],
                    "seed": row["seed"],
                    "ablation": row.get("ablation", "main"),
                    "ablation_value": row.get("ablation_value", ""),
                    "status": row.get("status", "pending"),
                    "attempt": row.get("attempt", 0),
                }
            )


def _planned(run_id, model, seed, status="pending", **extra):
    return {
        "run_id": run_id,
        "model": model,
        "seed": seed,
        "status": status,
        **extra,
    }


def test_final_ten_percent_and_auc(tmp_path):
    run_dir = _run(tmp_path / "suite", "comm_attention", 0, list(range(20)))
    result = compute_run_result(run_dir)
    assert result["final_window_points"] == 2
    assert result["mean_final_return"] == pytest.approx(18.5)
    assert result["normalized_return_auc"] == pytest.approx(9.5)
    assert result["comm_message_bits"] == 32


def test_bootstrap_is_deterministic():
    first = bootstrap_mean_ci([1, 2, 3, 4, 5], samples=1_000, seed=7)
    second = bootstrap_mean_ci([1, 2, 3, 4, 5], samples=1_000, seed=7)
    assert first == second


def test_analysis_writes_seed_aggregate_and_failed_report(tmp_path):
    suite = tmp_path / "suite"
    _run(suite, "comm_identity", 0, list(range(20)))
    _run(suite, "comm_identity", 1, list(range(1, 21)))
    _run(suite, "comm_attention", 0, [], status="failed")

    outputs = analyze_suite(suite, plots=True, bootstrap_samples=500)

    summary = outputs.summary_csv.read_text(encoding="utf-8")
    failures = outputs.failed_runs_csv.read_text(encoding="utf-8")
    assert "n_seeds" in summary
    assert "comm_identity" in summary
    assert "comm_attention-0" in failures
    assert "learning_curves.png" in {path.name for path in outputs.plot_files}
    assert "wall_clock.png" in {path.name for path in outputs.plot_files}


def test_manifest_inventory_reports_pending_failed_and_zero_completion_models(tmp_path):
    suite = tmp_path / "suite"
    identity = _run(suite, "comm_identity", 0, [1.0])
    attention = _run(suite, "comm_attention", 0, [], status="failed")
    _manifest(
        suite,
        [
            _planned(identity.name, "comm_identity", 0, "completed"),
            _planned("identity-1", "comm_identity", 1),
            _planned(attention.name, "comm_attention", 0, "failed"),
            _planned("attention-1", "comm_attention", 1),
            _planned("graph-0", "comm_graph", 0),
            _planned("graph-1", "comm_graph", 1),
        ],
    )

    per_run, summaries, unavailable = aggregate_suite(suite, bootstrap_samples=200)

    assert len(per_run) == 6
    assert {(row["run_id"], row["status"]) for row in per_run} >= {
        ("identity-1", "pending"),
        (attention.name, "failed"),
        ("graph-0", "pending"),
    }
    by_model = {row["model"]: row for row in summaries}
    assert by_model["comm_identity"]["n_expected_seeds"] == 2
    assert by_model["comm_identity"]["missing_expected_seeds"] == "1"
    assert by_model["comm_attention"]["n_completed_seeds"] == 0
    assert by_model["comm_attention"]["n_failed_runs"] == 1
    assert by_model["comm_graph"]["n_completed_seeds"] == 0
    assert by_model["comm_graph"]["mean_final_return"] is None
    assert by_model["comm_graph"]["missing_matched_seeds"] == "0;1"
    assert {row["status"] for row in unavailable} == {"pending", "failed"}
    assert len(unavailable) == 5


def test_paired_comparisons_use_manifest_expected_seeds_and_completed_pairs(tmp_path):
    suite = tmp_path / "suite"
    completed = []
    for model, seed, value in (
        ("comm_identity", 0, 1.0),
        ("comm_identity", 1, 2.0),
        ("comm_attention", 0, 4.0),
        ("comm_attention", 1, 6.0),
    ):
        completed.append(_run(suite, model, seed, [value]))
    plans = [
        _planned(run_dir.name, model, seed, "completed")
        for run_dir, (model, seed) in zip(
            completed,
            (
                ("comm_identity", 0),
                ("comm_identity", 1),
                ("comm_attention", 0),
                ("comm_attention", 1),
            ),
            strict=True,
        )
    ]
    plans.extend(
        [
            _planned("identity-2", "comm_identity", 2),
            _planned("attention-2", "comm_attention", 2),
        ]
    )
    _manifest(suite, plans)

    outputs = analyze_suite(suite, plots=False, bootstrap_samples=2_000, bootstrap_seed=9)
    assert outputs.paired_comparisons_csv is not None
    with outputs.paired_comparisons_csv.open(encoding="utf-8", newline="") as file:
        comparisons = list(csv.DictReader(file))

    assert {row["metric"] for row in comparisons} == {
        "mean_final_return",
        "normalized_return_auc",
    }
    final = next(row for row in comparisons if row["metric"] == "mean_final_return")
    assert final["direction"] == "model_minus_reference"
    assert final["reference_model"] == "comm_identity"
    assert final["n_expected_pairs"] == "3"
    assert final["expected_pair_seeds"] == "0;1;2"
    assert final["n_pairs"] == "2"
    assert final["paired_seeds"] == "0;1"
    assert final["missing_pair_seeds"] == "2"
    assert float(final["mean_paired_difference"]) == pytest.approx(3.5)
    assert float(final["difference_ci95_low"]) == pytest.approx(3.0)
    assert float(final["difference_ci95_high"]) == pytest.approx(4.0)
    assert float(final["paired_effect_size_cohens_dz"]) == pytest.approx(3.5 / (0.5**0.5))


def test_auc_comm_efficiency_and_ablation_plots_use_aggregate_ci_fields(tmp_path):
    suite = tmp_path / "ablation_suite"
    for value, bits in (("8", 64), ("16", 128)):
        for seed in (0, 1):
            _run(
                suite,
                "comm_attention",
                seed,
                [float(seed), float(seed + int(value))],
                ablation="message_dim",
                ablation_value=value,
                comm_bits=bits,
            )

    outputs = analyze_suite(suite, plots=True, bootstrap_samples=500)
    plot_names = {path.name for path in outputs.plot_files}
    assert "sample_efficiency.png" in plot_names
    assert "communication_efficiency.png" in plot_names
    assert "return_vs_message_dim.png" in plot_names

    with outputs.summary_csv.open(encoding="utf-8", newline="") as file:
        summaries = list(csv.DictReader(file))
    assert summaries
    for row in summaries:
        assert row["normalized_auc_ci95_low"]
        assert row["normalized_auc_ci95_high"]
        assert row["comm_message_bits_ci95_low"]
        assert row["comm_message_bits_ci95_high"]
