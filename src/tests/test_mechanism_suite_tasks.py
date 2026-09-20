"""Real PCP and synthetic-grid integration instruments, not scientific outcomes."""

from pathlib import Path
from benchmarl.experiment.callback import Callback


import pytest

from commstudy.experiments.mechanism_suite import METHODS, make_spec
from commstudy.experiments.config import experiment_spec_from_dict
from commstudy.experiments.runner import build_experiment
from commstudy.analysis.mechanism_suite import evaluate_pcp
from commstudy.analysis.mapdn_paired import evaluate_mapdn_paired
from test_mapdn_paired import config, _bank  # noqa: F401
from test_mapdn_integration import mapdn_config  # noqa: F401


class Initial(Callback):
    def on_setup(self):
        self.weights = {
            k: v.detach().clone()
            for k, v in self.experiment.group_policies["agents"].state_dict().items()
        }


@pytest.mark.parametrize("method", METHODS)
def test_real_pcp_all_methods_pair_and_replay(method, tmp_path):
    raw = make_spec("pcp", "radius1", method, 900, "smoke", tmp_path, Path("/unused"))
    raw["experiment"].update(save_folder=str(tmp_path / "scratch"), loggers=[], create_json=False)
    experiment = build_experiment(experiment_spec_from_dict(raw))
    try:
        result = evaluate_pcp(experiment, [3000000, 3000001], local_null=method in METHODS[:2])
        assert result == evaluate_pcp(
            experiment, [3000000, 3000001], local_null=method in METHODS[:2]
        )
        assert result["summaries"]["live"]["episode_steps"] == 100
        if method in METHODS[:2]:
            assert result["exact_local_null"]
        if method in METHODS[2:]:
            stats = result["arms"]["live"][0]["module_stats"]
            assert (
                stats["realized_sender_bits_per_step"]
                == 3 * (64 if method in METHODS[4:] else 32) * 32
            )
    finally:
        experiment.close()


@pytest.mark.parametrize("method", METHODS)
def test_synthetic_mapdn_all_methods_are_used_in_paired_evaluator(method, config, tmp_path):  # noqa: F811
    raw = make_spec("mapdn", "case33", method, 900, "smoke", tmp_path, Path("/unused"))
    raw["task_config"]["params"] = config
    raw["experiment"].update(save_folder=str(tmp_path / "scratch"), loggers=[], create_json=False)
    experiment = build_experiment(experiment_spec_from_dict(raw))
    try:
        bank = _bank(config, count=2)
        result = evaluate_mapdn_paired(
            experiment, bank, require_exact_local_null=method in METHODS[:2]
        )
        assert result == evaluate_mapdn_paired(
            experiment, bank, require_exact_local_null=method in METHODS[:2]
        )
        assert result["summaries"]["live"]["transition_count"] == 8
        assert experiment.task.state_spec(experiment.test_env) is None
    finally:
        experiment.close()


@pytest.mark.parametrize("method", METHODS)
def test_synthetic_mapdn_optimization_native_reload_and_serialized_closure(
    method,
    config,  # noqa: F811
    tmp_path,
):
    import json
    from commstudy.experiments.bookkeeping import RunContext, atomic_write_json
    from commstudy.experiments.runner import run_managed_experiment
    from commstudy.experiments.mechanism_execution import reload_closed_policy
    from commstudy.analysis.mechanism_suite import verify_suite_checkpoint

    raw = make_spec("mapdn", "case33", method, 900, "smoke", tmp_path, Path("/unused"))
    raw["task_config"]["params"] = config
    raw["experiment"].update(
        max_n_frames=16,
        checkpoint_interval=16,
        on_policy_collected_frames_per_batch=16,
        on_policy_minibatch_size=8,
        on_policy_n_minibatch_iters=1,
        loggers=["csv"],
        create_json=False,
    )
    context = RunContext("synthetic_instrument_only", method, tmp_path)
    callback = Initial()
    from commstudy.experiments.mechanism_suite import ROOT

    experiment = run_managed_experiment(
        experiment_spec_from_dict(raw), context, repo_root=ROOT, callbacks=[callback]
    )
    try:
        current = experiment.group_policies["agents"].state_dict()
        assert any(
            not __import__("torch").equal(v, current[k]) for k, v in callback.weights.items()
        )
        assert experiment.total_frames == 16
    finally:
        experiment.close()
    native = list(context.run_dir.glob("benchmarl/*/checkpoints/checkpoint_16.pt"))
    assert len(native) == 1
    summary = tmp_path / "training_summary.json"
    atomic_write_json(summary, {"run_dir": str(context.run_dir)})
    closure = json.loads(json.dumps({"training_summary_path": str(summary)}))
    frozen = reload_closed_policy(closure["training_summary_path"], tmp_path / "strict_reload")
    try:
        assert verify_suite_checkpoint(frozen, native[0], 16)["actor_critic_restored"]
        bank = _bank(config, count=2)
        result = evaluate_mapdn_paired(frozen, bank, require_exact_local_null=method in METHODS[:2])
        assert result["pairing_checks_passed"]
    finally:
        frozen.close()
