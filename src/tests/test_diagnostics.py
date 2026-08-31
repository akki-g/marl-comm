from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from commstudy.analysis import audit_run, finiteness_report, metric_trace
from commstudy.analysis.diagnostics import (
    _named_leaves,
    load_frozen_experiment,
    policy_action_diagnostics,
    rebuild_spec,
)
from commstudy.experiments.metrics import TidyMetricsWriter


BASE_OVERRIDES = [
    "algorithm=mappo",
    "task=vmas_simple_spread",
    "model=comm_identity",
    "seed=0",
    "experiment.max_n_frames=6000",
    "experiment.gamma=0.9",
    "algorithm_config.params.entropy_coef=0.1",
    "experiment.on_policy_n_minibatch_iters=5",
]


def _write_run(
    run_dir,
    *,
    evaluation_returns=(-900.0, -700.0, -500.0),
    entropies=(1.3, 0.8, 0.5),
    inject_nonfinite=False,
    status="completed",
    overrides=BASE_OVERRIDES,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "suite_id": run_dir.parent.name,
                "status": status,
                "task": "vmas_simple_spread",
                "algorithm": "mappo",
                "model": "comm_identity",
                "seed": 0,
                "overrides": list(overrides),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    writer = TidyMetricsWriter(run_dir)
    for index, (value, entropy) in enumerate(zip(evaluation_returns, entropies, strict=True)):
        frames = (index + 1) * 6_000
        writer.write(
            frames=frames,
            iteration=index,
            phase="evaluation",
            metrics={"return_mean": value},
        )
        # Per-episode samples must never be mistaken for evaluation points.
        writer.write(
            frames=frames,
            iteration=index,
            phase="evaluation",
            metrics={"return_episode": value - 5.0},
            sample=0,
        )
        writer.write(
            frames=frames,
            iteration=index,
            phase="training",
            group="agents",
            metrics={"entropy": entropy, "loss_critic": 100.0 - index},
        )
    if inject_nonfinite:
        writer.write(
            frames=24_000,
            iteration=3,
            phase="training",
            group="agents",
            metrics={"entropy": float("nan"), "loss_critic": float("inf")},
        )
    return run_dir


def test_finiteness_report_accepts_a_clean_run(tmp_path):
    run_dir = _write_run(tmp_path / "suite" / "clean")

    report = finiteness_report(run_dir)

    assert report["all_finite"] is True
    assert report["nonfinite_values"] == 0
    assert report["nonfinite_metrics"] == {}
    assert report["first_nonfinite_frames"] is None
    assert report["total_values"] > 0


def test_finiteness_report_locates_nonfinite_values(tmp_path):
    run_dir = _write_run(tmp_path / "suite" / "broken", inject_nonfinite=True)

    report = finiteness_report(run_dir)

    assert report["all_finite"] is False
    assert report["nonfinite_values"] == 2
    assert report["nonfinite_metrics"] == {
        "training/entropy": 1,
        "training/loss_critic": 1,
    }
    assert report["first_nonfinite_frames"] == 24_000


def test_metric_trace_reports_endpoints_and_extremes(tmp_path):
    run_dir = _write_run(
        tmp_path / "suite" / "traced",
        entropies=(1.3, -0.4, 0.9),
    )

    trace = metric_trace(run_dir, "training", "entropy")

    assert trace is not None
    assert (trace.first, trace.last) == (1.3, 0.9)
    assert (trace.minimum, trace.maximum) == (-0.4, 1.3)
    assert trace.first_frames == 6_000
    assert trace.last_frames == 18_000
    assert trace.nonfinite_count == 0


def test_metric_trace_ignores_per_episode_samples(tmp_path):
    run_dir = _write_run(tmp_path / "suite" / "samples")

    trace = metric_trace(run_dir, "evaluation", "return_mean")

    assert trace is not None
    assert trace.count == 3
    assert trace.last == -500.0


def test_metric_trace_returns_none_for_absent_metric(tmp_path):
    run_dir = _write_run(tmp_path / "suite" / "absent")

    assert metric_trace(run_dir, "training", "not_logged") is None


def test_metric_trace_counts_but_excludes_nonfinite_values(tmp_path):
    run_dir = _write_run(tmp_path / "suite" / "mixed", inject_nonfinite=True)

    trace = metric_trace(run_dir, "training", "entropy")

    assert trace is not None
    assert trace.nonfinite_count == 1
    assert trace.count == 4
    assert trace.last == 0.5
    assert math.isfinite(trace.minimum)


def test_audit_run_without_scratch_root_skips_the_policy_rollout(tmp_path):
    run_dir = _write_run(tmp_path / "suite" / "metrics-only")

    report = audit_run(run_dir, config_root=tmp_path, scratch_root=None)

    assert report["policy_audited"] is False
    assert report["status"] == "completed"
    assert report["evaluation_points"] == 3
    assert report["final_evaluation_return"] == -500.0
    assert report["entropy_first"] == 1.3
    assert report["entropy_last"] == 0.5
    assert "deterministic_saturated_action_fraction" not in report


def test_audit_run_skips_the_policy_rollout_without_a_checkpoint(tmp_path, config_root):
    run_dir = _write_run(tmp_path / "suite" / "no-checkpoint")

    report = audit_run(run_dir, config_root=config_root, scratch_root=tmp_path / "scratch")

    assert report["policy_audited"] is False


def test_named_leaves_skips_next_and_non_float_entries():
    from tensordict import TensorDict

    batch = TensorDict(
        {
            "agents": TensorDict(
                {"action": torch.zeros(2, 3), "info": torch.zeros(2, 3, dtype=torch.int64)},
                batch_size=[2],
            ),
            "next": TensorDict(
                {"agents": TensorDict({"action": torch.ones(2, 3)}, batch_size=[2])},
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )

    found = _named_leaves(batch, "action")

    assert len(found) == 1
    assert torch.equal(found[0], torch.zeros(6, dtype=torch.float64))


def test_rebuild_spec_reproduces_the_recorded_overrides(tmp_path, config_root):
    run_dir = _write_run(tmp_path / "suite" / "spec")

    spec = rebuild_spec(run_dir, config_root)

    assert spec.model == "comm_identity"
    assert spec.experiment["gamma"] == 0.9
    assert spec.experiment["on_policy_n_minibatch_iters"] == 5
    assert spec.algorithm_config["params"]["entropy_coef"] == 0.1


@pytest.mark.parametrize("exploration", ["DETERMINISTIC", "RANDOM"])
def test_policy_action_diagnostics_on_a_real_frozen_actor(tmp_path, config_root, exploration):
    """Build a real VMAS/MAPPO actor, freeze it, and measure its actions."""

    from commstudy.experiments.config import load_experiment_spec
    from commstudy.experiments.runner import build_experiment

    spec = load_experiment_spec(config_root, [*BASE_OVERRIDES, "experiment.evaluation=false"])
    source = build_experiment(
        dataclasses.replace(
            spec,
            experiment={**spec.experiment, "save_folder": str(tmp_path / "source")},
        )
    )
    run_dir = _write_run(tmp_path / "suite" / "real")
    (run_dir / "checkpoints").mkdir()
    torch.save(
        {
            "group_policies": {
                group: policy.state_dict()
                for group, policy in source.group_policies.items()
            }
        },
        run_dir / "checkpoints" / "policy_state.pt",
    )
    source.test_env.close()

    experiment = load_frozen_experiment(
        run_dir,
        config_root=config_root,
        scratch_root=tmp_path / "scratch",
    )
    try:
        diagnostics = policy_action_diagnostics(
            experiment,
            exploration=exploration,
            steps=5,
            seed=0,
        )
    finally:
        experiment.test_env.close()

    assert diagnostics.exploration == exploration
    assert diagnostics.action_scalars > 0
    assert diagnostics.finite_action_fraction == 1.0
    assert 0.0 <= diagnostics.saturated_action_fraction <= 1.0
    assert diagnostics.max_abs_action <= 1.0
    # BenchMARL's TanhNormal policy exposes the raw distribution parameters,
    # which is exactly the saturation signal this audit exists to catch.
    assert diagnostics.mean_scale is not None
    assert diagnostics.mean_scale > 0.0
    assert diagnostics.mean_abs_location is not None
