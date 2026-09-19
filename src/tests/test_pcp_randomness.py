"""Scientific reset/RNG invariants on the real PCP task factory."""

import dataclasses
import random
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from commstudy.experiments import build_experiment, load_experiment_spec
from commstudy.experiments.evaluation import EvaluationIsolatedExperiment
from commstudy.communication.channel import DropoutChannel
from commstudy.tasks import resolve_task
from commstudy.utils.rng import RNGState, preserve_rng_state


def _env(seed=73, worlds=3, max_steps=100):
    task = resolve_task("vmas_predator_capture_prey", {"max_steps": max_steps})
    return closing(task.get_env_fun(worlds, True, seed, "cpu")())


def _scenario(env):
    return env.base_env.scenario


def _step(env, td):
    td = td.clone()
    td.update(env.full_action_spec.zero())
    return env.step(td)["next"]


def _assert_rng_equal(left, right):
    assert left.python == right.python
    assert left.numpy[0] == right.numpy[0]
    assert np.array_equal(left.numpy[1], right.numpy[1])
    assert left.numpy[2:] == right.numpy[2:]
    assert torch.equal(left.torch_cpu, right.torch_cpu)
    if left.torch_cuda is not None:
        assert len(left.torch_cuda) == len(right.torch_cuda)
        assert all(
            torch.equal(a, b) for a, b in zip(left.torch_cuda, right.torch_cuda, strict=True)
        )
    if left.vmas is not None:
        assert torch.equal(left.vmas[0], right.vmas[0])
        assert np.array_equal(left.vmas[1][1], right.vmas[1][1])
        assert left.vmas[1][2:] == right.vmas[1][2:]
        assert left.vmas[2] == right.vmas[2]


def test_seeded_reset_matches_fresh_environment_after_arbitrary_history():
    with _env() as used, _env() as fresh:
        used.set_seed(991)
        td = used.reset()
        for _ in range(7):
            td = _step(used, td)
        scenario = _scenario(used)
        scenario.good_agents()[0].wander_dir.fill_(0.5)
        # Cross a schedule-block boundary independently of physical history.
        for _ in range(270):
            scenario._scripted_prey_action(scenario.good_agents()[0])
        used.set_seed(991)
        fresh.set_seed(991)
        left, right = used.reset(), fresh.reset()
        assert torch.equal(left["state"], right["state"])
        assert not scenario.good_agents()[0].wander_dir.any()
        assert not scenario.exogenous_steps.any()
        for _ in range(5):
            left, right = _step(used, left), _step(fresh, right)
            assert torch.equal(left["state"], right["state"])
            assert torch.equal(left["adversary", "reward"], right["adversary", "reward"])


def test_partial_reset_changes_only_selected_world_and_preserves_other_noise():
    with _env() as left_env, _env() as right_env:
        left, right = left_env.reset(), right_env.reset()
        left_scenario, right_scenario = _scenario(left_env), _scenario(right_env)
        for _ in range(3):
            left, right = _step(left_env, left), _step(right_env, right)
        before = left["state"].clone()
        episodes = left_scenario.exogenous_episode_ids.clone()
        left = left_env.reset(left.clone().set("_reset", torch.tensor([[0], [1], [0]]).bool()))
        assert torch.equal(left["state"][[0, 2]], before[[0, 2]])
        assert not left_scenario.good_agents()[0].wander_dir[1].any()
        assert left_scenario.exogenous_steps[1].eq(0).all()
        assert left_scenario.exogenous_steps[[0, 2]].eq(3).all()
        assert torch.equal(left_scenario.exogenous_episode_ids[[0, 2]], episodes[[0, 2]])
        assert left_scenario.exogenous_episode_ids[1] != episodes[1]
        # Carry the unaffected worlds past the next noise-block boundary.
        for _ in range(270):
            a = left_scenario._scripted_prey_action(left_scenario.good_agents()[0])
            b = right_scenario._scripted_prey_action(right_scenario.good_agents()[0])
            assert torch.equal(a[[0, 2]], b[[0, 2]])


def test_channel_or_actor_draws_do_not_change_prey_disturbances():
    with _env() as left_env, _env() as right_env:
        left_env.reset()
        right_env.reset()
        left, right = _scenario(left_env), _scenario(right_env)
        for _ in range(270):
            torch.rand(131)
            torch.randn(47)
            a = left._scripted_prey_action(left.good_agents()[0])
            torch.rand(83)
            b = right._scripted_prey_action(right.good_agents()[0])
            assert torch.equal(a, b)


def test_environment_construction_seed_reset_and_step_preserve_caller_rng():
    # Import VMAS before capture so its class-shared stream exists on both sides.
    resolve_task("vmas_predator_capture_prey", {})
    before = RNGState.capture()
    with _env() as env:
        env.set_seed(44)
        td = env.reset()
        _step(env, td)
    _assert_rng_equal(before, RNGState.capture())


def test_rng_context_restores_all_streams_after_exception():
    before = RNGState.capture()
    with pytest.raises(RuntimeError, match="deliberate"):
        with preserve_rng_state(seed=921):
            random.random()
            np.random.rand()
            torch.randn(6)
            with _env() as env:
                _step(env, env.reset())
            raise RuntimeError("deliberate")
    _assert_rng_equal(before, RNGState.capture())


@pytest.mark.parametrize("static", [True, False])
def test_evaluation_channel_stream_is_independent_of_collection_and_advances_when_nonstatic(static):
    channel = DropoutChannel(p=0.5, mode="evaluation")
    experiment = object.__new__(EvaluationIsolatedExperiment)
    experiment.policy = torch.nn.Sequential(channel)
    experiment.config = SimpleNamespace(evaluation_static=static)
    experiment.seed = 81
    messages = torch.ones(64, 3, 2)

    with experiment._evaluation_context():
        first = channel(messages).sender_mask.clone()
    assert channel._rng == {}
    # Even arbitrary prior collection draws cannot alter a static evaluation;
    # non-static evaluation advances a separate stream and remains reproducible.
    channel.mode = "always"
    channel(messages)
    training_state = channel._rng["cpu"].get_state().clone()
    with experiment._evaluation_context():
        second = channel(messages).sender_mask.clone()
    assert torch.equal(training_state, channel._rng["cpu"].get_state())
    assert torch.equal(first, second) is static
    experiment._evaluation_rng_index = 1
    with experiment._evaluation_context():
        repeated = channel(messages).sender_mask.clone()
    assert torch.equal(second, repeated)


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("raises", [False, True])
def test_inserted_framework_evaluation_preserves_next_training_transition(
    config_root, tmp_path, static, raises, monkeypatch
):
    spec = load_experiment_spec(
        config_root,
        [
            "task=vmas_predator_capture_prey",
            "model=pcp_comm_identity",
            "critic_model=pcp_critic",
            "task_config.params.max_steps=7",
            "experiment.evaluation_episodes=2",
            "experiment.on_policy_n_envs_per_worker=3",
            f"experiment.evaluation_static={str(static).lower()}",
            "experiment.evaluation_deterministic_actions=false",
            "experiment.render=false",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
        ],
    )
    spec = dataclasses.replace(spec, experiment={**spec.experiment, "save_folder": str(tmp_path)})
    experiment = build_experiment(spec)
    try:
        with closing(experiment.env_func()) as train, closing(experiment.env_func()) as reference:
            left, right = train.reset(), reference.reset()
            left, right = _step(train, left), _step(reference, right)
            assert torch.equal(left["state"], right["state"])
            before = RNGState.capture()
            if raises:

                def fail_after_evaluation(rollouts):
                    raise RuntimeError("evaluation callback failed")

                monkeypatch.setattr(experiment, "_on_evaluation_end", fail_after_evaluation)
                with pytest.raises(RuntimeError, match="evaluation callback failed"):
                    experiment.evaluate()
            else:
                experiment._evaluation_loop()
            _assert_rng_equal(before, RNGState.capture())
            left, right = _step(train, left), _step(reference, right)
            assert torch.equal(left["state"], right["state"])
            assert torch.equal(left["adversary", "reward"], right["adversary", "reward"])
    finally:
        experiment.close()
