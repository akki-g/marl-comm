from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from commstudy.analysis import audit_run, finiteness_report, metric_trace
from commstudy.analysis.diagnostics import (
    _named_leaves,
    load_frozen_experiment,
    policy_action_diagnostics,
    rebuild_spec,
)
from commstudy.experiments.metrics import TidyMetricsWriter
from commstudy.experiments.config import (
    experiment_spec_from_dict,
    load_experiment_spec,
    resolved_experiment_dict,
    scientific_config_sha256,
)
from commstudy.experiments.bookkeeping import capture_versions, task_runtime_contract
from commstudy.experiments.provenance import source_fingerprint


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
    spec=None,
    experiment=None,
    training_group="agents",
):
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    if spec is None:
        spec = load_experiment_spec(repo_root / "configs", list(overrides))
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create({"commstudy": resolved_experiment_dict(spec)})),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "suite_id": run_dir.parent.name,
                "status": status,
                "task": spec.task,
                "algorithm": spec.algorithm,
                "model": spec.model,
                "seed": spec.seed,
                "overrides": list(overrides),
                "versions": capture_versions(),
                "source_sha256": source_fingerprint(repo_root),
                "scientific_config_sha256": scientific_config_sha256(spec),
                "runtime": {
                    key: spec.experiment.get(key)
                    for key in ("sampling_device", "train_device", "buffer_device")
                },
                "task_runtime_contract": task_runtime_contract(
                    experiment or SimpleNamespace(task=None)
                ),
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
            group=training_group,
            metrics={"entropy": entropy, "loss_critic": 100.0 - index},
        )
    if inject_nonfinite:
        writer.write(
            frames=24_000,
            iteration=3,
            phase="training",
            group=training_group,
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

    found = _named_leaves(batch, "action", "agents")

    assert len(found) == 1
    assert torch.equal(found[0], torch.zeros(6, dtype=torch.float64))


def test_rebuild_spec_reproduces_snapshot_without_current_defaults(tmp_path, config_root):
    run_dir = _write_run(tmp_path / "suite" / "spec")

    metadata = json.loads((run_dir / "metadata.json").read_text())
    metadata["overrides"] = ["algorithm_config.params.entropy_coef=999", "experiment.gamma=0.5"]
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    spec = rebuild_spec(run_dir, tmp_path / "nonexistent-current-configs")

    assert spec.model == "comm_identity"
    assert spec.experiment["gamma"] == 0.9
    assert spec.experiment["on_policy_n_minibatch_iters"] == 5
    assert spec.algorithm_config["params"]["entropy_coef"] == 0.1


def test_rebuild_spec_rejects_missing_snapshot_instead_of_replaying_old_overrides(tmp_path):
    run = _write_run(tmp_path / "run")
    (run / "resolved_config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="resolved specification"):
        rebuild_spec(run)


def test_audit_infers_saved_pcp_group_and_never_mixes_prey_training_metrics(tmp_path, config_root):
    spec = load_experiment_spec(
        config_root,
        ["task=vmas_predator_capture_prey", "model=pcp_comm_identity", "critic_model=pcp_critic"],
    )
    run = _write_run(tmp_path / "pcp", spec=spec, training_group="adversary")
    writer = TidyMetricsWriter(run)
    writer.write(
        frames=24_000,
        iteration=4,
        phase="training",
        group="agent",
        metrics={"entropy": -999.0, "loss_critic": float("nan")},
    )
    report = audit_run(run)
    assert report["measured_group"] == "adversary"
    assert report["entropy_last"] == 0.5
    assert report["entropy_min"] == 0.5
    assert report["all_finite"] is True
    assert report["all_groups_finiteness"]["all_finite"] is False
    with pytest.raises(ValueError, match="multiple groups"):
        metric_trace(run, "training", "entropy")


def test_audit_rejects_ambiguous_groups_without_saved_selection(tmp_path):
    run = _write_run(tmp_path / "ambiguous")
    snapshot = run / "resolved_config.yaml"
    document = OmegaConf.load(snapshot)
    document.commstudy.task_config.pop("return_groups", None)
    OmegaConf.save(document, snapshot)
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["scientific_config_sha256"] = scientific_config_sha256(
        experiment_spec_from_dict(OmegaConf.to_container(document.commstudy, resolve=True))
    )
    metadata_path.write_text(json.dumps(metadata))
    TidyMetricsWriter(run).write(
        frames=24_000,
        iteration=4,
        phase="training",
        group="second",
        metrics={"entropy": -99},
    )
    with pytest.raises(ValueError, match="explicit measured group"):
        audit_run(run)


def test_real_pcp_action_audit_counts_32_episodes_and_excludes_poisoned_prey(config_root, tmp_path):
    from commstudy.experiments.runner import build_experiment

    spec = load_experiment_spec(
        config_root,
        [
            "task=vmas_predator_capture_prey",
            "model=pcp_comm_identity",
            "critic_model=pcp_critic",
            "experiment.evaluation_episodes=5",
            "experiment.evaluation=false",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
            "task_config.params.max_steps=5",
        ],
    )
    spec = dataclasses.replace(spec, experiment={**spec.experiment, "save_folder": str(tmp_path)})
    experiment = build_experiment(spec)

    class PoisonPrey(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, td):
            result = self.policy(td)
            for name, value in (("loc", math.nan), ("scale", math.inf), ("action", 1.0)):
                result.set(("agent", name), torch.full_like(result["agent", name], value))
            return result

    try:
        assert experiment.test_env.batch_size == torch.Size([5])
        with pytest.raises(ValueError, match="explicit group"):
            policy_action_diagnostics(experiment, episodes=32, steps=2)
        experiment.policy = PoisonPrey(experiment.policy)
        result = policy_action_diagnostics(experiment, group="adversary", episodes=32, steps=2)
        assert result.episodes == 32
        assert len(set(result.episode_ids)) == 32
        assert result.episode_transitions == (2,) * 32
        assert result.action_scalars == 32 * 2 * 3 * 2
        assert result.finite_action_fraction == result.finite_location_fraction == 1.0
        assert result.finite_scale_fraction == 1.0
        assert result.max_abs_action < 0.999
        assert result.complete_episodes == 0
    finally:
        experiment.close()


@pytest.fixture
def frozen_run(tmp_path, config_root):
    from commstudy.experiments.runner import build_experiment

    spec = load_experiment_spec(config_root, [*BASE_OVERRIDES, "experiment.evaluation=false"])
    source = build_experiment(
        dataclasses.replace(
            spec, experiment={**spec.experiment, "save_folder": str(tmp_path / "source")}
        )
    )
    try:
        run = _write_run(tmp_path / "run", spec=spec, experiment=source)
        (run / "checkpoints").mkdir()
        torch.save(
            {
                "group_policies": {
                    group: policy.state_dict() for group, policy in source.group_policies.items()
                }
            },
            run / "checkpoints" / "policy_state.pt",
        )
    finally:
        source.close()
    return run


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_checkpoint_requires_exact_policy_group_set(frozen_run, tmp_path, mutation):
    checkpoint_path = frozen_run / "checkpoints" / "policy_state.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if mutation == "missing":
        checkpoint["group_policies"].pop("agents")
    else:
        checkpoint["group_policies"]["unexpected"] = {}
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="group set"):
        load_frozen_experiment(frozen_run, scratch_root=tmp_path / "audit")


def test_checkpoint_runtime_mismatch_requires_explicit_recorded_acknowledgement(
    frozen_run, tmp_path
):
    metadata_path = frozen_run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"] = "historical-source"
    metadata["versions"]["vmas"] = "historical-version"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source/runtime"):
        load_frozen_experiment(frozen_run, scratch_root=tmp_path / "audit")
    before = {
        path.relative_to(frozen_run): path.read_bytes()
        for path in frozen_run.rglob("*")
        if path.is_file()
    }
    experiment = load_frozen_experiment(
        frozen_run,
        scratch_root=tmp_path / "audit",
        allow_runtime_mismatch=True,
        analysis_overrides={
            "sampling_device": "cpu",
            "train_device": "cpu",
            "buffer_device": "cpu",
        },
    )
    try:
        assert experiment.analysis_provenance["allow_runtime_mismatch"] is True
        assert set(experiment.analysis_provenance["compatibility_mismatches"]) == {
            "source_sha256",
            "versions.vmas",
        }
        assert experiment.analysis_provenance["analysis_overrides"]["sampling_device"] == "cpu"
    finally:
        experiment.close()
    assert before == {
        path.relative_to(frozen_run): path.read_bytes()
        for path in frozen_run.rglob("*")
        if path.is_file()
    }


def test_checkpoint_rejects_changed_semantic_contract(frozen_run, tmp_path):
    path = frozen_run / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["task_runtime_contract"]["evaluation_randomness_protocol"] = "legacy"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="task runtime contract"):
        load_frozen_experiment(frozen_run, scratch_root=tmp_path / "audit")


def test_checkpoint_rejects_changed_scientific_snapshot(frozen_run, tmp_path):
    path = frozen_run / "resolved_config.yaml"
    document = OmegaConf.load(path)
    document.commstudy.experiment.gamma = 0.91
    OmegaConf.save(document, path)
    with pytest.raises(ValueError, match="scientific configuration hash"):
        load_frozen_experiment(frozen_run, scratch_root=tmp_path / "audit")


def test_checkpoint_rejects_scratch_inside_run_and_scientific_analysis_overrides(
    frozen_run, tmp_path
):
    with pytest.raises(ValueError, match="outside the audited run"):
        load_frozen_experiment(frozen_run, scratch_root=frozen_run / "audit")
    with pytest.raises(ValueError, match="Unsupported analysis-only"):
        load_frozen_experiment(
            frozen_run, scratch_root=tmp_path / "audit", analysis_overrides={"gamma": 0.99}
        )


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
    run_dir = _write_run(tmp_path / "suite" / "real", spec=spec, experiment=source)
    (run_dir / "checkpoints").mkdir()
    torch.save(
        {
            "group_policies": {
                group: policy.state_dict() for group, policy in source.group_policies.items()
            }
        },
        run_dir / "checkpoints" / "policy_state.pt",
    )
    source.close()

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
        experiment.close()

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
    assert diagnostics.group == "agents"
    assert diagnostics.episodes == 5
    assert diagnostics.action_scalars == 5 * 5 * 3 * 2
    assert len(diagnostics.episode_ids) == 5
    assert diagnostics.complete_episodes == 0
    assert diagnostics.scale_quantiles["p05"] <= diagnostics.scale_quantiles["p95"]
