"""Exercise the real managed PCP measurement path without a study launch."""

from __future__ import annotations

import csv
import hashlib
import json
import math

import pytest
import torch
from benchmarl.experiment.callback import Callback

from commstudy.analysis.diagnostics import load_frozen_experiment, policy_action_diagnostics
from commstudy.analysis.pcp_domain import policy_domain_audit
from commstudy.communication.identity import IdentityComm
from commstudy.experiments import load_experiment_spec
from commstudy.experiments.bookkeeping import RunContext
from commstudy.experiments.runner import run_managed_experiment
from commstudy.tasks.vmas.domain_metrics import PCP_DOMAIN_PROTOCOL


class _CaptureEvaluation(Callback):
    def __init__(self):
        super().__init__()
        self.evaluations = []

    def on_evaluation_end(self, rollouts):
        self.evaluations.append([rollout.clone() for rollout in rollouts])


def _metric(rows, phase, name, *, group="adversary", sample=""):
    selected = [row for row in rows if row["phase"] == phase
                and row["metric"] == name and row["group"] == group
                and row["sample"] == str(sample)]
    assert len(selected) == 1, (phase, name, group, sample, selected)
    return float(selected[0]["value"])


def _artifact_hashes(run_dir):
    return {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest() for name in (
        "metadata.json", "resolved_config.yaml", "status.json", "summary.json",
        "metrics.csv", "checkpoints/policy_state.pt",
    )}


def test_managed_pcp_domains_and_frozen_audits_count_episodes_once(config_root, tmp_path):
    """One real 6k update, three batched evaluations, and strict final-policy reload.

    The three predators carry duplicate team info. Neither callback aggregation
    nor the frozen single-world seed bank may turn three episodes into nine.
    All generated artifacts are temporary and labelled as validation evidence.
    """
    spec = load_experiment_spec(config_root, [
        "task=vmas_predator_capture_prey",
        "model=pcp_comm_identity",
        "critic_model=pcp_critic",
        "seed=7",
        "experiment.max_n_frames=6000",
        "experiment.on_policy_collected_frames_per_batch=6000",
        "experiment.on_policy_n_envs_per_worker=10",
        "experiment.on_policy_n_minibatch_iters=1",
        "experiment.evaluation=true",
        "experiment.evaluation_interval=6000",
        "experiment.evaluation_episodes=3",
        "experiment.evaluation_deterministic_actions=true",
        "experiment.render=false",
        "experiment.loggers=[]",
        # Upstream BenchMARL evaluates only with logging or JSON enabled.
        "experiment.create_json=true",
        "experiment.checkpoint_at_end=false",
    ])
    context = RunContext("pcp_domain_validation", "identity_seed007", tmp_path / "runs")
    captured = _CaptureEvaluation()
    experiment = run_managed_experiment(
        spec, context, repo_root=config_root.parent, callbacks=[captured], validation_run=True,
    )
    assert experiment.total_frames == 6000
    assert experiment.n_iters_performed == 1
    assert len(captured.evaluations) == 1
    assert len(captured.evaluations[0]) == 3
    with (context.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert all(math.isfinite(float(row["value"])) for row in rows)
    assert _metric(rows, "collection_domain", "transition_count") == 6000
    assert _metric(rows, "collection_domain", "reward_accounting_max_abs_error") == 0
    assert _metric(rows, "evaluation_domain", "episode_count") == 3
    assert _metric(rows, "evaluation_domain", "complete_episodes") == 3
    assert _metric(rows, "evaluation_domain", "transition_count") == 300
    assert {row["sample"] for row in rows if row["phase"] == "evaluation_domain_episode"} == {
        "0", "1", "2",
    }
    returns = []
    for index, rollout in enumerate(captured.evaluations[0]):
        assert rollout.batch_size == torch.Size([100])
        reward = rollout["next", "adversary", "reward"]
        pairs = rollout["next", "adversary", "info", "contact_pairs"]
        assert reward.shape == pairs.shape == (100, 3, 1)
        assert torch.equal(reward, 10 * pairs)
        assert not rollout["next", "terminated"].any()
        assert not rollout["next", "done"][:-1].any()
        assert rollout["next", "truncated"][-1].all()
        # Read one team copy, with no dependency on the domain reducer itself.
        episode_return = float(reward[:, 0, 0].sum())
        returns.append(episode_return)
        assert _metric(rows, "evaluation_domain_episode", "return", sample=index) == (
            episode_return
        )
        assert _metric(rows, "evaluation", "return_episode", group="", sample=index) == (
            episode_return
        )
        for name, expected in (("episode_steps", 100), ("complete", 1),
                               ("truncated", 1), ("terminated", 0),
                               ("reward_accounting_max_abs_error", 0)):
            assert _metric(rows, "evaluation_domain_episode", name, sample=index) == expected
    expected_return = sum(returns) / 3
    assert _metric(rows, "evaluation_domain", "return") == pytest.approx(expected_return)
    assert _metric(rows, "evaluation", "return_mean") == pytest.approx(expected_return)
    assert _metric(rows, "evaluation", "return_mean", group="") == pytest.approx(
        expected_return
    )
    for phase, transitions in (("collection", 6000), ("evaluation", 300)):
        assert _metric(rows, phase, "health_transition_count") == transitions
        for leaf in ("action", "loc", "scale"):
            assert _metric(rows, phase, f"{leaf}_finite_fraction") == 1
            assert _metric(rows, phase, f"{leaf}_scalar_count") == transitions * 3 * 2
    assert _metric(rows, "collection", "replay_transition_count") == 6000
    assert _metric(rows, "collection", "replay_log_prob_max_abs_error") <= 1e-5
    assert _metric(rows, "training", "optimization_transition_count") == 6000
    assert _metric(rows, "training", "advantage_finite_fraction") == 1
    assert _metric(rows, "training", "value_target_finite_fraction") == 1

    metadata = json.loads(context.metadata_path.read_text())
    assert metadata["status"] == "completed"
    assert metadata["execution_purpose"] == "validation"
    assert metadata["protocol"] is None
    before = _artifact_hashes(context.run_dir)
    frozen = load_frozen_experiment(context.run_dir, scratch_root=tmp_path / "frozen")
    try:
        assert any(isinstance(module, IdentityComm)
                   for module in frozen.group_policies["adversary"].modules())
        audit = policy_domain_audit(frozen, episodes=3, seed=9000)
        assert audit["domain_protocol"] == PCP_DOMAIN_PROTOCOL
        provenance = audit["analysis_provenance"]
        assert provenance["source_run"] == str(context.run_dir.resolve())
        assert provenance["source_sha256"] == metadata["source_sha256"]
        assert provenance["scientific_config_sha256"] == metadata["scientific_config_sha256"]
        assert provenance["checkpoint_sha256"] == before["checkpoints/policy_state.pt"]
        assert provenance["task_runtime_contract"] == metadata["task_runtime_contract"]
        assert provenance["compatibility_mismatches"] == {}
        assert provenance["allow_runtime_mismatch"] is False
        assert set(audit["modes"]) == {"DETERMINISTIC", "RANDOM"}
        for mode, result in audit["modes"].items():
            assert result["episode_count"] == result["complete_episodes"] == 3
            assert result["transition_count"] == 300
            assert result["reward_accounting_max_abs_error"] == 0
            assert [episode["seed"] for episode in result["episodes"]] == [9000, 9001, 9002]
            for episode in result["episodes"]:
                metrics = episode["metrics"]
                assert metrics["episode_steps"] == 100
                assert metrics["complete"] == metrics["truncated"] == 1
                assert metrics["terminated"] == 0
                assert metrics["return"] == 10 * metrics["contact_pair_steps"]
            actions = policy_action_diagnostics(
                frozen, group="adversary", exploration=mode, episodes=3, seed=9000,
            )
            assert actions.episodes == actions.complete_episodes == 3
            assert actions.episode_transitions == (100, 100, 100)
            assert actions.action_scalars == actions.finite_action_scalars == 1800
            assert actions.finite_location_fraction == actions.finite_scale_fraction == 1
    finally:
        frozen.close()
    assert _artifact_hashes(context.run_dir) == before
