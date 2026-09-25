"""Synthetic feeder invariants; real-data gates are separate explicit CLI runs."""

from helpers import example_spec

import dataclasses

import numpy as np
import pytest
import torch
from torchrl.envs import SerialEnv, check_env_specs

from commstudy.tasks import resolve_task
from commstudy.adapters.mapdn import (
    TaskConfig,
    _make_adapter,
    make_mapdn_env,
    PowerGridTaskClass,
)
from commstudy.tasks.mapdn_data import build_split_manifest, save_split_manifest


@pytest.fixture(scope="module")
def mapdn_config(tmp_path_factory):
    pytest.importorskip("pandapower")
    pytest.importorskip("pettingzoo")
    import mapdn_fixture as fixture

    data = tmp_path_factory.mktemp("mapdn_integration")
    fixture._build_network(data)
    # Real MAPDN CSVs share numeric load identifiers across P/Q profiles.
    # Retain the user's fixture physics while matching that source schema.
    import pandas as pd

    for name in ("load_active.csv", "load_reactive.csv"):
        frame = pd.read_csv(data / name)
        frame.columns = ["time", *map(str, range(frame.shape[1] - 1))]
        frame.to_csv(data / name, index=False)
    manifest = build_split_manifest(data, max_steps=4, history=2)
    manifest_path = data / "manifest.json"
    save_split_manifest(manifest, manifest_path)
    return dataclasses.asdict(
        TaskConfig(
            data_path=str(data),
            manifest_path=str(manifest_path),
            max_steps=4,
            history=2,
            measurement_noise=False,
        )
    )


def zero_actions(env):
    return {agent: np.zeros(1, np.float32) for agent in env.possible_agents}


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_steps": 0},
        {"history": True},
        {"solver_algorithm": "unknown"},
        {"action_scale": 2.0},
        {"evaluation_split": "train"},
        {"split": "oops"},
        {"split": "validation"},
        {"split": "test"},
        {"voltage_weight": -1.0},
        {"q_weight": 0.1, "line_weight": 0.1},
        {"state_space": ["vm_pu", "vm_pu"]},
        {"unknown_option": 1},
    ],
)
def test_invalid_task_contract_rejected(overrides):
    with pytest.raises((ValueError, TypeError)):
        resolve_task("mapdn_voltage_control", overrides)


def test_registry_resolves_strict_defaults_without_data():
    task = resolve_task("mapdn_voltage_control")
    assert isinstance(task, PowerGridTaskClass)
    assert task.config["evaluation_split"] == "validation"


def test_spec_history_and_terminal_diagnostics(mapdn_config):
    env = make_mapdn_env(mapdn_config, seed=1)
    try:
        check_env_specs(env)
        task = resolve_task("mapdn_voltage_control", mapdn_config)
        assert set(task.observation_spec(env).keys(True, True)) == {("agents", "observation")}
        assert task.state_spec(env) is None
        td = env.reset()
        obs = td["agents", "observation"]
        assert torch.equal(
            obs[:, : obs.shape[-1] // 2], torch.zeros_like(obs[:, : obs.shape[-1] // 2])
        )
        rollout = env.rollout(4)
        assert rollout["next", "truncated"][-1].all()
        assert not rollout["next", "terminated"].any()
        assert rollout["next", "agents", "info", "valid"].eq(1).all()
        metrics = task.log_info(rollout)
        assert metrics["mapdn/transitions"] == 4
        assert metrics["mapdn/bus_transition_count"] == 4 * 13
        assert metrics["mapdn/solver_failure_count"] == 0
    finally:
        env.close()


def test_action_reward_and_next_observation_rows_are_aligned(mapdn_config):
    env = _make_adapter(mapdn_config, seed=5)
    raw = env._env
    try:
        first = raw._start_rows()[0]
        env.reset(seed=5, options={"start_row": first})
        assert raw._row == first
        initial = raw.powergrid.sgen.p_mw.to_numpy(copy=True)
        assert np.allclose(initial, raw.pv_data.iloc[first].to_numpy())
        initial_obs, _ = env.reset(seed=5, options={"start_row": first})
        obs, rewards, _, _, infos = env.step(zero_actions(env))
        diagnostic = env.get_diagnostics()
        assert diagnostic["reward_row"] == first
        assert diagnostic["observation_row"] == first + 1
        assert diagnostic["reward"] == rewards["agent_0"]
        assert all(not x for x in infos.values())
        assert np.allclose(raw.powergrid.sgen.p_mw, raw.pv_data.iloc[first + 1])
        half = env.obs_dim // 2
        assert np.array_equal(obs["agent_0"][:half], initial_obs["agent_0"][half:])
        # Public observations reflect a solved next state, not a stale bus table.
        import pandapower as pp

        before = raw.powergrid.res_bus.vm_pu.to_numpy(copy=True)
        pp.runpp(raw.powergrid, **raw.solver_kwargs)
        assert np.allclose(before, raw.powergrid.res_bus.vm_pu, atol=1e-10)
    finally:
        env.close()


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.zeros(2), np.array([np.inf])])
def test_nonfinite_or_wrong_shape_actions_fail_before_physics(mapdn_config, bad):
    env = _make_adapter(mapdn_config)
    try:
        actions = zero_actions(env)
        actions["agent_0"] = bad
        row = env._env._row
        with pytest.raises(ValueError):
            env.step(actions)
        assert env._env._row == row
    finally:
        env.close()


def test_failure_is_terminal_and_preserved_in_torchrl_batch(mapdn_config, monkeypatch):
    env = make_mapdn_env(mapdn_config)
    try:
        td = env.reset()
        adapter = env.base_env._env
        monkeypatch.setattr(adapter._env, "_solve", lambda: False)
        td = env.rand_action(td)
        transition = env.step(td)
        next_td = transition["next"]
        assert next_td["terminated"].all()
        assert not next_td["truncated"].any()
        assert next_td["agents", "info", "destroy"].eq(1).all()
        assert next_td["agents", "info", "valid"].eq(1).all()
        assert next_td["agents", "info", "voltage_metrics_valid"].eq(0).all()
        assert torch.isfinite(next_td["agents", "observation"]).all()
        assert next_td["agents", "reward"].le(-200).all()
        metrics = PowerGridTaskClass.log_info(transition)
        assert metrics["mapdn/transitions"] == 1
        assert metrics["mapdn/voltage_observed_transitions"] == 0
        assert metrics["mapdn/bus_transition_count"] == 0
        assert metrics["mapdn/solver_failure_count"] == 1
        assert "mapdn/percentage_of_v_out_of_control" not in metrics
    finally:
        env.close()


def test_reset_failure_does_not_retry_or_replace_row(mapdn_config, monkeypatch):
    from commstudy.environments.mapdn.research import ResetPowerFlowError

    env = _make_adapter(mapdn_config)
    calls = []
    try:
        monkeypatch.setattr(env._env, "_solve", lambda: calls.append(1) or False)
        with pytest.raises(ResetPowerFlowError):
            env.reset(seed=1)
        assert len(calls) == 1
    finally:
        env.close()


def test_advance_failure_keeps_valid_action_voltage_and_failure_count(mapdn_config, monkeypatch):
    env = make_mapdn_env(mapdn_config)
    try:
        td = env.reset()
        raw = env.base_env._env._env
        solve = raw._solve
        count = 0

        def fail_advance():
            nonlocal count
            count += 1
            return solve() if count == 1 else False

        monkeypatch.setattr(raw, "_solve", fail_advance)
        transition = env.step(env.rand_action(td))
        metrics = PowerGridTaskClass.log_info(transition)
        assert transition["next", "terminated"].all()
        assert metrics["mapdn/voltage_observed_transitions"] == 1
        assert metrics["mapdn/solver_failure_count"] == 1
        assert metrics["mapdn/advance_solver_failure"] == 1
        assert "mapdn/percentage_of_v_out_of_control" in metrics
    finally:
        env.close()


def test_inverter_actions_follow_declared_agent_order(mapdn_config, monkeypatch):
    env = _make_adapter(mapdn_config)
    try:
        env.reset(seed=1)
        raw = env._env
        active = raw.powergrid.sgen.p_mw.to_numpy(copy=True)
        fractions = np.linspace(-0.4, 0.4, env.num_mapdn_agents, dtype=np.float32)
        expected = np.sqrt(np.maximum(raw.s_max**2 - active**2, 0)) * fractions
        recorded = []
        solve = raw._solve

        def record():
            recorded.append(raw.powergrid.sgen.q_mvar.to_numpy(copy=True))
            return solve()

        monkeypatch.setattr(raw, "_solve", record)
        actions = {
            a: np.array([fractions[i]], np.float32)
            for i, a in reversed(list(enumerate(env.possible_agents)))
        }
        env.step(actions)
        assert np.allclose(recorded[0], expected, rtol=0, atol=1e-12)
    finally:
        env.close()


def test_runtime_contract_binds_vendor_source_and_normalizer(mapdn_config, monkeypatch, tmp_path):
    from commstudy.environments import mapdn
    from commstudy.tasks.mapdn_data import load_split_manifest, TrainingNormalizer

    task = resolve_task("mapdn_voltage_control", mapdn_config)
    original = task.runtime_contract(None)
    assert original["mapdn_origin"]["repository_head"]
    assert original["normalization_sha256"] is None
    assert original["powergrid_versions"]["pandapower"] == "3.3.0"
    (tmp_path / "__init__.py").write_text("# changed source\n")
    monkeypatch.setattr(mapdn, "__file__", str(tmp_path / "__init__.py"))
    assert task.runtime_contract(None)["mapdn_source_sha256"] != original["mapdn_source_sha256"]
    manifest = load_split_manifest(mapdn_config["manifest_path"])
    normalization_path = tmp_path / "normalization.json"
    normalizer = TrainingNormalizer.fit(
        np.array([[1.0], [2.0]]), split="train", manifest_sha256=manifest["sha256"]
    )
    normalizer.save(normalization_path)
    task.config["normalization_path"] = str(normalization_path)
    assert task.runtime_contract(None)["normalization_sha256"] == normalizer.sha256


def test_training_noise_scales_and_private_rng_survive_sibling_steps(mapdn_config):
    from commstudy.tasks.mapdn_data import load_split_manifest

    config = {**mapdn_config, "measurement_noise": True}
    manifest = load_split_manifest(config["manifest_path"])
    first, second, sibling = (_make_adapter(config, seed=7) for _ in range(3))
    try:
        row = manifest["splits"]["train"]["starts"][17]
        for env in (first, second):
            env.reset(seed=7, options={"start_row": row})
        sibling.reset(seed=98)
        assert np.array_equal(
            first._env.pv_std, np.asarray(manifest["training_statistics"]["pv_std"]) / 100
        )
        np.random.random(19)
        for _ in range(3):
            a = first.step(zero_actions(first))
            sibling.step(zero_actions(sibling))
            b = second.step(zero_actions(second))
            for agent in first.possible_agents:
                assert np.array_equal(a[0][agent], b[0][agent])
            assert a[1:4] == b[1:4]
            assert first.get_diagnostics() == second.get_diagnostics()
    finally:
        first.close()
        second.close()
        sibling.close()


def test_normalizer_from_another_split_manifest_is_rejected(mapdn_config, tmp_path):
    from commstudy.tasks.mapdn_data import TrainingNormalizer

    path = tmp_path / "foreign.json"
    TrainingNormalizer.fit(np.ones((2, 50)), split="train", manifest_sha256="0" * 64).save(path)
    with pytest.raises(ValueError, match="different data manifest"):
        _make_adapter({**mapdn_config, "normalization_path": str(path)})


@pytest.mark.parametrize("changed", [{"pv_scale": 2.0}, {"measurement_noise": True}])
def test_normalization_calibration_settings_cannot_silently_change(mapdn_config, tmp_path, changed):
    from commstudy.tasks.mapdn_data import TrainingNormalizer

    settings = {
        k: v
        for k, v in mapdn_config.items()
        if k not in {"data_path", "manifest_path", "normalization_path"}
    }
    manifest = build_split_manifest(mapdn_config["data_path"], 4, 2, settings=settings)
    path = tmp_path / "manifest.json"
    save_split_manifest(manifest, path)
    normalization_path = tmp_path / "normalizer.json"
    TrainingNormalizer.fit(
        np.ones((2, 44)), split="train", manifest_sha256=manifest["sha256"]
    ).save(normalization_path)
    with pytest.raises(ValueError, match="settings do not match"):
        _make_adapter(
            {
                **mapdn_config,
                **changed,
                "manifest_path": str(path),
                "normalization_path": str(normalization_path),
            }
        )


def test_serial_env_seeds_and_metrics_survive_parallel_transport(mapdn_config):
    task = resolve_task("mapdn_voltage_control", mapdn_config)
    factory = task.get_env_fun(2, True, 3, "cpu")
    env = SerialEnv(2, factory)
    try:
        env.set_seed(7)
        check_env_specs(env)
        batch = env.rollout(4)
        assert batch.batch_size == (2, 4)
        assert task.log_info(batch)["mapdn/transitions"] == 8
    finally:
        env.close()


def test_managed_experiment_uses_validation_only_for_test_env(mapdn_config, config_root, tmp_path):
    from commstudy.experiments import build_experiment

    spec = example_spec(config_root, task="mapdn")
    spec = dataclasses.replace(
        spec,
        task_config={"params": mapdn_config},
        experiment={
            **spec.experiment,
            "max_n_frames": 8,
            "on_policy_collected_frames_per_batch": 8,
            "on_policy_n_envs_per_worker": 1,
            "on_policy_minibatch_size": 8,
            "on_policy_n_minibatch_iters": 1,
            "evaluation": False,
            "loggers": [],
            "create_json": False,
            "save_folder": str(tmp_path / "model"),
        },
    )
    experiment = build_experiment(spec)
    try:
        assert experiment.test_env.base_env._env._env.split == "validation"
        params_before = [p.detach().clone() for p in experiment.policy.parameters()]
        experiment.run()
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, experiment.policy.parameters(), strict=True)
        )
    finally:
        experiment.close()
