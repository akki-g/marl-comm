"""A validation label cannot fall back to BenchMARL's full training budget."""

from dataclasses import replace
import json
from unittest.mock import MagicMock

import pytest

from commstudy.experiments.bookkeeping import RunContext
from commstudy.experiments.config import load_experiment_spec
from commstudy.experiments.runner import run_managed_experiment


def _spec(config_root):
    return load_experiment_spec(config_root, [
        "task=vmas_predator_capture_prey", "model=pcp_comm_identity", "critic_model=pcp_critic",
    ])


@pytest.mark.parametrize("limit", ["omitted", None, True, False, 6001, 0, -1])
def test_validation_budget_must_be_explicit_positive_and_bounded(
    config_root, tmp_path, monkeypatch, limit,
):
    original = _spec(config_root)
    experiment_config = dict(original.experiment)
    if limit == "omitted":
        experiment_config.pop("max_n_frames")
    else:
        experiment_config["max_n_frames"] = limit
    spec = replace(original, experiment=experiment_config)
    context = RunContext(suite_id="validation_fixture", run_id="invalid_budget",
                         output_root=tmp_path)
    build = MagicMock(side_effect=AssertionError("An invalid budget must not construct a model"))
    monkeypatch.setattr("commstudy.experiments.runner.build_experiment", build)
    with pytest.raises((ValueError, TypeError)):
        run_managed_experiment(spec, context, repo_root=tmp_path, validation_run=True)
    build.assert_not_called()
    assert not context.run_dir.exists()


def test_explicit_6000_frame_validation_budget_passes_without_scientific_training(
    config_root, tmp_path, monkeypatch,
):
    original = _spec(config_root)
    spec = replace(original, experiment={**original.experiment, "max_n_frames": 6000})
    context = RunContext(suite_id="validation_fixture", run_id="bounded_budget",
                         output_root=tmp_path)
    experiment, recorder = MagicMock(), MagicMock()
    monkeypatch.setattr("commstudy.experiments.runner.RunRecorder", lambda *a, **kw: recorder)
    build = MagicMock(return_value=experiment)
    monkeypatch.setattr("commstudy.experiments.runner.build_experiment", build)

    assert run_managed_experiment(
        spec, context, repo_root=tmp_path, validation_run=True,
    ) is experiment
    assert build.call_args.args[0].experiment["max_n_frames"] == 6000
    experiment.run.assert_called_once_with()
    recorder.complete.assert_called_once_with(experiment)
    assert json.loads(context.metadata_path.read_text())["execution_purpose"] == "validation"
