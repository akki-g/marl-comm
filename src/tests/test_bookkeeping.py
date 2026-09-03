from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from commstudy.experiments.bookkeeping import (
    RunAlreadyCompletedError,
    RunContext,
    RunRecorder,
    capture_git_state,
    make_run_id,
    retry_context,
    summarize_metrics_csv,
)
from commstudy.experiments.metrics import TidyMetricsWriter
from commstudy.experiments import load_experiment_spec


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "Tests")
    (path / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "add", "tracked.py")
    _git(path, "commit", "-qm", "initial")
    return path


class FakeExperiment:
    def __init__(self, folder: Path) -> None:
        self.group_policies = {"agents": nn.Linear(3, 2)}
        self.losses = {
            "agents": SimpleNamespace(critic_network=nn.Linear(4, 1))
        }
        self.config = SimpleNamespace(
            sampling_device="cpu", train_device="cpu", buffer_device="cpu"
        )
        self.algorithm_config = SimpleNamespace(clip_epsilon=0.2)
        self.task = SimpleNamespace(config={"n_agents": 3})
        self.model_config = SimpleNamespace(hidden_dim=8)
        self.critic_model_config = SimpleNamespace(hidden_dim=8)
        self.folder_name = folder
        self.total_frames = 12_000
        self.n_iters_performed = 2
        self.total_time = 3.5


def test_descriptive_run_id_and_retry(tmp_path):
    context = RunContext(
        suite_id="suite",
        run_id=make_run_id(
            task="vmas_simple_spread",
            algorithm="mappo",
            model="comm_attention",
            seed=3,
        ),
        output_root=tmp_path,
    )
    assert context.run_id == "simple_spread__mappo__attention__seed003"
    context.run_dir.mkdir(parents=True)
    first_retry = retry_context(context)
    assert first_retry.run_id.endswith("__retry01")
    assert first_retry.retry_of == context.run_id


def test_dirty_patch_includes_untracked_sources(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "new_module.py").write_text("ANSWER = 42\n", encoding="utf-8")
    run_dir = tmp_path / "evidence"
    run_dir.mkdir()

    state = capture_git_state(repo, run_dir)

    assert state["dirty"] is True
    assert "new_module.py" in state["untracked_files"]
    patch = (run_dir / "git.patch").read_text(encoding="utf-8")
    assert "new_module.py" in patch
    assert "ANSWER = 42" in patch


def test_recorder_writes_atomic_lifecycle_and_policy_checkpoint(
    tmp_path, config_root
):
    repo = _repo(tmp_path / "repo")
    spec = load_experiment_spec(config_root, ["model=comm_identity"])
    context = RunContext("suite", "run", tmp_path / "outputs")
    recorder = RunRecorder(context, spec, repo_root=repo)
    recorder.start()

    fake = FakeExperiment(context.benchmarl_dir / "generated")
    recorder.record_experiment(fake)
    writer = TidyMetricsWriter(context.run_dir)
    for index in range(20):
        writer.write(
            frames=(index + 1) * 6_000,
            iteration=index,
            phase="evaluation",
            metrics={"return_mean": float(index)},
        )
    summary = recorder.complete(fake)

    metadata = json.loads(context.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["parameters"]["actor_total"] == 8
    assert metadata["parameters"]["critic_total"] == 5
    assert summary["mean_final_return"] == pytest.approx(18.5)
    checkpoint = context.run_dir / metadata["policy_checkpoint"]
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["seed"] == spec.seed
    assert payload["frames"] == fake.total_frames
    assert "agents" in payload["group_policies"]

    with pytest.raises(RunAlreadyCompletedError):
        RunRecorder(context, spec, repo_root=repo).start()


def test_recorder_preserves_failure(tmp_path, config_root):
    repo = _repo(tmp_path / "repo")
    spec = load_experiment_spec(config_root)
    context = RunContext("suite", "failed", tmp_path / "outputs")
    recorder = RunRecorder(context, spec, repo_root=repo)
    recorder.start()
    error = RuntimeError("deliberate failure")
    recorder.fail(error)

    status = json.loads(context.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error"]["type"] == "RuntimeError"
    assert "deliberate failure" in status["error"]["message"]


def test_tidy_metrics_has_stable_long_schema(tmp_path):
    writer = TidyMetricsWriter(tmp_path)
    writer.write(
        frames=6_000,
        iteration=0,
        phase="training",
        group="agents",
        metrics={"loss_objective": torch.tensor(1.25), "entropy": 0.5},
    )
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["metric"] for row in rows] == ["loss_objective", "entropy"]
    assert all(row["frames"] == "6000" for row in rows)


def test_summary_ignores_per_group_returns_on_a_two_group_task(tmp_path):
    """The run summary must read the ungrouped study figure, not average it
    with the per-group scalars logged beside it.

    PCP logs three ``return_mean`` rows per evaluation: the study figure (no
    group) plus one per BenchMARL group, and its two groups are exactly
    zero-sum. Averaging all three divides the predators' return by three, which
    is how the first pilot reported 2.33 for a run that scored 7.0. A one-group
    task cannot catch this -- there the extra row equals the study figure.
    """
    writer = TidyMetricsWriter(tmp_path)
    for index in range(10):
        frames = (index + 1) * 6_000
        predator = float(index)
        writer.write(
            frames=frames, iteration=index, phase="evaluation",
            metrics={"return_mean": predator},
        )
        writer.write(
            frames=frames, iteration=index, phase="evaluation",
            metrics={"return_mean": predator}, group="adversary",
        )
        writer.write(
            frames=frames, iteration=index, phase="evaluation",
            metrics={"return_mean": -predator}, group="agent",
        )

    summary = summarize_metrics_csv(tmp_path / "metrics.csv")

    assert summary["evaluation_points"] == 10
    assert summary["mean_final_return"] == pytest.approx(9.0)
    assert summary["mean_final_return"] == pytest.approx(
        compute_run_result_curve_final(tmp_path)
    )


def compute_run_result_curve_final(run_dir: Path) -> float:
    """The analysis path's own answer, so the two stay pinned together."""
    from commstudy.analysis.aggregate import _evaluation_curve, _metric_rows

    curve = _evaluation_curve(_metric_rows(run_dir / "metrics.csv"))
    return curve[-1][1]
