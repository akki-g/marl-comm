from __future__ import annotations

import csv
import json

import pytest

from commstudy.analysis.report import render_report, write_report


def _write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(
    suite_dir,
    run_id,
    *,
    status="completed",
    dirty=False,
    device="cpu",
    commit="abc123",
    gpu=None,
):
    run_dir = suite_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = {"train_device": device}
    if gpu is not None:
        runtime["cuda_device_name"] = gpu
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": suite_dir.name,
                "status": status,
                "task": "vmas_simple_spread",
                "algorithm": "mappo",
                "model": "comm_broadcast",
                "seed": 0,
                "frames": 600000,
                "git": {"commit_sha": commit, "dirty": dirty},
                "runtime": runtime,
                "versions": {"torch": "2.13.0", "benchmarl": "1.5.2"},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


@pytest.fixture
def suite(tmp_path):
    suite_dir = tmp_path / "runs" / "demo_suite"
    results_dir = tmp_path / "results" / "demo_suite"
    suite_dir.mkdir(parents=True)
    _run(suite_dir, "demo__broadcast__seed000")

    _write_csv(
        results_dir / "demo_suite_per_run.csv",
        [
            {
                "run_id": "demo__broadcast__seed000",
                "status": "completed",
                "model": "comm_broadcast",
                "seed": "0",
                "ablation": "main",
                "ablation_value": "",
                "mean_final_return": "-445.1",
            },
            {
                "run_id": "demo__identity__seed000",
                "status": "completed",
                "model": "comm_identity",
                "seed": "0",
                "ablation": "main",
                "ablation_value": "",
                "mean_final_return": "-449.2",
            },
            {
                "run_id": "demo__graph__seed000",
                "status": "failed",
                "model": "comm_graph",
                "seed": "0",
                "ablation": "main",
                "ablation_value": "",
                "mean_final_return": "",
            },
        ],
    )
    _write_csv(
        results_dir / "demo_suite_summary.csv",
        [
            {
                "model": "comm_broadcast",
                "ablation": "main",
                "ablation_value": "",
                "n_seeds": "1",
                "mean_final_return": "-445.1",
                "ci95_low": "-450.0",
                "ci95_high": "-440.0",
                "mean_normalized_auc": "-538.7",
                "actor_params": "27140",
                "comm_params": "8192",
                "mean_comm_realized_bits_per_step": "6144",
                "n_saliency_seeds": "1",
                "mean_saliency_return_with_comm": "-407.5",
                "mean_saliency_return_without_comm": "-578.3",
                "mean_saliency_return_delta": "170.8",
                "saliency_return_delta_ci95_low": "170.8",
                "saliency_return_delta_ci95_high": "170.8",
                "mean_saliency_return_delta_fraction": "0.295",
                "mean_saliency_action_shift_mean": "0.166",
            },
            {
                "model": "comm_attention",
                "ablation": "message_dim",
                "ablation_value": "8",
                "n_seeds": "3",
                "mean_final_return": "-500.0",
                "ci95_low": "-520.0",
                "ci95_high": "-480.0",
                "mean_normalized_auc": "-560.0",
            },
            {
                "model": "comm_attention",
                "ablation": "message_dim",
                "ablation_value": "64",
                "n_seeds": "3",
                "mean_final_return": "-470.0",
                "ci95_low": "-490.0",
                "ci95_high": "-450.0",
                "mean_normalized_auc": "-540.0",
            },
        ],
    )
    _write_csv(
        results_dir / "demo_suite_failed_runs.csv",
        [
            {
                "run_id": "demo__graph__seed000",
                "status": "failed",
                "error_type": "RuntimeError",
                "error_message": "cuda out of memory",
            }
        ],
    )
    _write_csv(
        results_dir / "demo_suite_paired_comparisons.csv",
        [
            {
                "model": "comm_broadcast",
                "reference_model": "comm_identity",
                "ablation": "main",
                "ablation_value": "",
                "metric": "mean_final_return",
                "n_pairs": "1",
                "mean_paired_difference": "4.1",
                "difference_ci95_low": "4.1",
                "difference_ci95_high": "4.1",
                "paired_effect_size_cohens_dz": "",
            }
        ],
    )
    return suite_dir, results_dir


def test_report_states_inventory_and_warns_when_incomplete(suite):
    suite_dir, results_dir = suite

    report = render_report(suite_dir, results_dir)

    assert "## Run inventory" in report
    assert "completed | 2" in report
    assert "failed | 1" in report
    assert "rows are not completed" in report


def test_report_includes_saliency_with_controls_and_interpretation(suite):
    suite_dir, results_dir = suite

    report = render_report(suite_dir, results_dir)

    assert "## Communication saliency" in report
    assert "170.8" in report
    assert "changes behaviour without" in report


def test_report_orders_numeric_ablation_values_numerically(suite):
    suite_dir, results_dir = suite

    report = render_report(suite_dir, results_dir)
    section = report.split("### message dim")[1]

    assert section.index("| 8 |") < section.index("| 64 |")


def test_report_surfaces_failures_rather_than_hiding_them(suite):
    suite_dir, results_dir = suite

    report = render_report(suite_dir, results_dir)

    assert "## Failed and unavailable rows" in report
    assert "cuda out of memory" in report
    assert "demo__graph__seed000" in report


def test_report_records_dirty_tree_and_device(suite, tmp_path):
    suite_dir, results_dir = suite
    _run(suite_dir, "dirty_run", dirty=True, device="cuda", commit="def456")

    report = render_report(suite_dir, results_dir)

    assert "dirty" in report
    assert "cuda" in report
    # Two commits and two devices in one suite is a comparability hazard.
    assert "did not all execute under one environment" in report


def test_report_warns_when_a_suite_mixes_gpu_models(suite):
    """V100 and H100 rows are not directly comparable; say so explicitly."""

    suite_dir, results_dir = suite
    _run(suite_dir, "v100_run", device="cuda", gpu="Tesla V100-PCIE-32GB")
    _run(suite_dir, "h100_run", device="cuda", gpu="NVIDIA H100 PCIe")

    report = render_report(suite_dir, results_dir)

    assert "GPU models" in report
    assert "Tesla V100-PCIE-32GB" in report
    assert "NVIDIA H100 PCIe" in report
    assert "--gres=gpu:h100:1" in report


def test_report_is_quiet_when_one_gpu_model_is_used_throughout(tmp_path):
    suite_dir = tmp_path / "runs" / "uniform"
    results_dir = tmp_path / "results" / "uniform"
    suite_dir.mkdir(parents=True)
    for name in ("a", "b"):
        _run(suite_dir, name, device="cuda", gpu="NVIDIA H100 PCIe")
    _write_csv(
        results_dir / "uniform_per_run.csv",
        [{"run_id": "a", "status": "completed", "model": "comm_graph", "seed": "0",
          "ablation": "main", "ablation_value": "", "mean_final_return": "-500"}],
    )
    _write_csv(results_dir / "uniform_summary.csv", [])

    report = render_report(suite_dir, results_dir)

    assert "did not all execute under one environment" not in report
    assert "NVIDIA H100 PCIe" in report


def test_report_notes_absent_saliency_instead_of_reporting_zero(tmp_path):
    suite_dir = tmp_path / "runs" / "bare"
    results_dir = tmp_path / "results" / "bare"
    suite_dir.mkdir(parents=True)
    _write_csv(
        results_dir / "bare_per_run.csv",
        [{"run_id": "a", "status": "completed", "model": "comm_graph", "seed": "0",
          "ablation": "main", "ablation_value": "", "mean_final_return": "-500"}],
    )
    _write_csv(
        results_dir / "bare_summary.csv",
        [{"model": "comm_graph", "ablation": "main", "ablation_value": "", "n_seeds": "1",
          "mean_final_return": "-500", "ci95_low": "-510", "ci95_high": "-490"}],
    )

    report = render_report(suite_dir, results_dir)

    assert "Not measured" in report
    assert "## Communication saliency" in report


def test_write_report_creates_file_and_lists_figures(suite):
    suite_dir, results_dir = suite
    (results_dir / "learning_curves.png").write_bytes(b"\x89PNG")

    path = write_report(suite_dir, results_dir)

    assert path.name == "REPORT.md"
    body = path.read_text(encoding="utf-8")
    assert "![Learning curves](learning_curves.png)" in body
    assert "## Artifacts" in body


def test_report_handles_a_suite_with_no_results_at_all(tmp_path):
    suite_dir = tmp_path / "runs" / "empty"
    suite_dir.mkdir(parents=True)

    report = render_report(suite_dir, tmp_path / "results" / "empty")

    assert "# empty" in report
    assert "_No rows._" in report
