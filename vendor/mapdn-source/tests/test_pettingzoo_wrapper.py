"""Guards for the MAPDN -> PettingZoo wrapper.

Most of these protect properties that would fail *silently* rather than raise:
information leaking across the partial-observability boundary, a corrupted
observation history, unenforced action bounds, or randomness that couples
separate environment instances. Each of those would change experimental
results without producing an error, so they are asserted rather than assumed.
"""
import numpy as np
import pytest

from mapdn.environments.pettingzoo import BENCHMARL_HAS_STATE, MAPDNParallelEnv

from .conftest import N_AGENTS


def make_env(config, **overrides):
    return MAPDNParallelEnv({**config, **overrides})


def zero_actions(env):
    return {agent: np.zeros(1, dtype=np.float32) for agent in env.agents}


# ---------------------------------------------------------------- API surface

def test_parallel_api(config):
    from pettingzoo.test import parallel_api_test

    parallel_api_test(make_env(config, episode_limit=12), num_cycles=40)


def test_agents_available_before_reset(config):
    env = make_env(config)
    assert env.agents == env.possible_agents
    assert len(env.agents) == N_AGENTS


def test_spaces_are_per_agent(config):
    env = make_env(config)
    assert env.observation_space("agent_0") is not env.observation_space("agent_1")
    assert env.action_space("agent_0").shape == (1,)
    # bounds come straight from action_scale / action_bias
    assert env.action_space("agent_0").low[0] == pytest.approx(-0.8)
    assert env.action_space("agent_0").high[0] == pytest.approx(0.8)


def test_group_map_matches_vmas_convention(config):
    # keeps BenchMARL model configs shared across the task family
    assert list(make_env(config).group_map) == ["agents"]


def test_max_steps_matches_actual_episode_length(config):
    env = make_env(config, episode_limit=8)
    env.reset(seed=0)
    steps = 0
    while env.agents:
        env.step(zero_actions(env))
        steps += 1
    assert steps == env.max_steps


def test_render_is_not_supported(config):
    with pytest.raises(NotImplementedError):
        make_env(config).render()


# ------------------------------------------------------- R1: no info leakage

def test_infos_are_empty_at_reset_and_step(config):
    """Agents must not receive MAPDN's whole-network aggregates.

    Every key MAPDN puts in `info` is computed across all buses/inverters, so
    handing them to each agent would be an always-on global channel present
    even in the no-comm baseline.
    """
    env = make_env(config)
    _, infos = env.reset(seed=0)
    assert all(info == {} for info in infos.values())

    _, _, _, _, infos = env.step(zero_actions(env))
    assert all(info == {} for info in infos.values())


def test_global_diagnostics_stay_out_of_band(config):
    env = make_env(config)
    env.reset(seed=0)
    env.step(zero_actions(env))
    diagnostics = env.get_diagnostics()
    assert "percentage_of_v_out_of_control" in diagnostics
    assert "total_line_loss" in diagnostics


def test_no_info_keys_reach_the_torchrl_observation_spec(config):
    """The leak would materialise here, as ('agents', 'info', ...) entries.

    TorchRL builds its info spec from the *reset* infos, so this also fails if
    reset starts returning populated infos while step does not.
    """
    torchrl = pytest.importorskip("torchrl")
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper

    env = make_env(config)
    wrapped = PettingZooWrapper(
        env, categorical_actions=False, group_map=env.group_map
    )
    keys = [str(key) for key in wrapped.observation_spec.keys(True, True)]
    assert not any("info" in key for key in keys), keys


def test_state_is_not_exposed_to_models_by_default(config):
    """Symmetry with VMAS, whose state_spec is None: the MAPPO critic must take
    concatenated per-agent observations on both members of the task family."""
    pytest.importorskip("torchrl")
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper

    assert BENCHMARL_HAS_STATE is False
    env = make_env(config)
    wrapped = PettingZooWrapper(
        env, categorical_actions=False, group_map=env.group_map
    )
    assert "state" not in [str(k) for k in wrapped.observation_spec.keys(True, True)]


# ------------------------------------------------- R2: RNG stays per-instance

def test_seeding_is_reproducible(config):
    def rollout(seed):
        env = make_env(config)
        env.reset(seed=seed)
        return [env.step(zero_actions(env))[0]["agent_0"].copy() for _ in range(3)]

    for first, second in zip(rollout(7), rollout(7)):
        assert np.allclose(first, second)


def test_dynamics_ignore_the_global_numpy_stream(config):
    def rollout(disturb):
        env = make_env(config)
        env.reset(seed=7)
        if disturb:
            np.random.rand(5)
        return [env.step(zero_actions(env))[0]["agent_0"].copy() for _ in range(3)]

    for undisturbed, disturbed in zip(rollout(False), rollout(True)):
        assert np.allclose(undisturbed, disturbed)


def test_sibling_environments_do_not_interfere(config):
    solo_env = make_env(config)
    solo_env.reset(seed=7)
    solo = [solo_env.step(zero_actions(solo_env))[0]["agent_0"].copy() for _ in range(3)]

    env, sibling = make_env(config), make_env(config)
    env.reset(seed=7)
    sibling.reset(seed=999)
    interleaved = []
    for _ in range(3):
        interleaved.append(env.step(zero_actions(env))[0]["agent_0"].copy())
        sibling.step(zero_actions(sibling))

    for expected, actual in zip(solo, interleaved):
        assert np.allclose(expected, actual)


def test_episodes_still_vary_without_a_seed(config):
    env = make_env(config)
    starts = set()
    for _ in range(6):
        env.reset()
        starts.add(
            (
                env._env._episode_start_day,
                env._env._episode_start_hour,
                env._env._episode_start_interval,
            )
        )
    assert len(starts) > 1


# --------------------------------------------- R3: observation history intact

def test_history_stacking_is_not_duplicated(config):
    """reset() must call get_obs() exactly once.

    MAPDN's get_obs() appends to the observation history on every call, so a
    second call at reset stacks a duplicated frame where the older slot should
    still be zero-padded.
    """
    env = make_env(config, history=2)
    observations, _ = env.reset(seed=0)

    assert len(env._env.obs_history[0]) == 1

    half = env.obs_dim // 2
    older, newer = observations["agent_0"][:half], observations["agent_0"][half:]
    assert np.allclose(older, 0.0)
    assert not np.allclose(older, newer)

    stepped = env.step(zero_actions(env))[0]["agent_0"]
    assert np.allclose(stepped[:half], newer)


# --------------------------------------------------- R5: config is pinned

@pytest.mark.parametrize(
    "overrides",
    [
        {"action_scale": None},
        {"action_bias": None},
        {"voltage_barrier_type": None},
        {"voltage_barrier_type": "not_a_barrier"},
        {"mode": "decentralised"},
        {"history": 0},
        {"episode_limit": 1},
    ],
)
def test_invalid_config_fails_at_the_wrapper_boundary(config, overrides):
    with pytest.raises((ValueError, TypeError)):
        make_env(config, **overrides)


def test_task_signature_records_what_defines_the_task(config):
    signature = make_env(config).task_signature
    assert signature["voltage_barrier_type"] == "l1"
    assert signature["action_low"] == pytest.approx(-0.8)
    assert signature["benchmarl_has_state"] is False


# ------------------------------------------------ S1: action bounds enforced

def test_out_of_range_actions_saturate(config):
    env = make_env(config)
    env.reset(seed=0)
    active = env._env.powergrid.sgen["p_mw"].to_numpy(copy=True)
    capacity = np.sqrt(env._env.s_max**2 - active**2)

    env.step({agent: np.array([5.0], np.float32) for agent in env.agents})

    reactive = env._env.powergrid.sgen["q_mvar"].to_numpy(copy=True)
    assert np.allclose(reactive, capacity * 0.8)


def test_bounds_are_enforced_in_the_underlying_env_too(config):
    env = make_env(config)
    env.reset(seed=0)
    active = env._env.powergrid.sgen["p_mw"].to_numpy(copy=True)
    capacity = np.sqrt(env._env.s_max**2 - active**2)

    env._env.step(np.full(N_AGENTS, -9.0))

    reactive = env._env.powergrid.sgen["q_mvar"].to_numpy(copy=True)
    assert np.allclose(reactive, capacity * -0.8)


def test_missing_action_raises_rather_than_misaligning(config):
    env = make_env(config)
    env.reset(seed=0)
    actions = zero_actions(env)
    del actions["agent_2"]
    with pytest.raises(KeyError):
        env.step(actions)


# ------------------------------------- MAPDN semantics that must not change

def test_actions_map_to_the_right_inverter(config):
    """A transposition here would not error, it would silently swap which agent
    controls which bus."""
    env = make_env(config)
    env.reset(seed=0)
    active = env._env.powergrid.sgen["p_mw"].to_numpy(copy=True)
    capacity = np.sqrt(env._env.s_max**2 - active**2)

    scale = [0.1 * (i + 1) for i in range(N_AGENTS)]
    env.step({f"agent_{i}": np.array([scale[i]], np.float32) for i in range(N_AGENTS)})

    reactive = env._env.powergrid.sgen["q_mvar"].to_numpy(copy=True)
    assert np.allclose(reactive, capacity * np.array(scale))


def test_observations_are_zone_local(config):
    env = make_env(config)
    observations, _ = env.reset(seed=0)
    topology = env.agent_topology

    same_zone = [
        (a, b)
        for a in env.possible_agents
        for b in env.possible_agents
        if a < b and topology[a]["zone"] == topology[b]["zone"]
    ]
    cross_zone = [
        (a, b)
        for a in env.possible_agents
        for b in env.possible_agents
        if a < b and topology[a]["zone"] != topology[b]["zone"]
    ]

    # agents sharing a zone see the same bus readings and differ only in their
    # own PV active power and their own reactive power
    for a, b in same_zone:
        differing = np.sum(~np.isclose(observations[a], observations[b]))
        assert differing == 2, (a, b, differing)

    # agents in different zones share no live observation component
    for a, b in cross_zone:
        shared = np.isclose(observations[a], observations[b]) & (observations[a] != 0.0)
        assert not shared.any(), (a, b)


def test_reward_is_a_shared_team_scalar(config):
    env = make_env(config)
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step(zero_actions(env))
    assert len(set(rewards.values())) == 1


def test_time_limit_truncates_rather_than_terminates(config):
    env = make_env(config, episode_limit=6)
    env.reset(seed=0)
    while env.agents:
        _, _, terminations, truncations, _ = env.step(zero_actions(env))
    assert all(truncations.values())
    assert not any(terminations.values())
