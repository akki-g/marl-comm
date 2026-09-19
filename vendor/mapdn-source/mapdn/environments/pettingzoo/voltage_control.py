from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.utils.env import ParallelEnv

from ..var_voltage_control.voltage_barrier.voltage_barrier_registry import (
    Voltage_Barrier,
)
from ..var_voltage_control.voltage_control_env import VoltageControl

# Whether the BenchMARL task built on this env should expose a global state to
# the centralised critic.
#
# Held at False on purpose. BenchMARL's VmasTask.state_spec returns None
# unconditionally, so MAPPO's critic consumes concatenated per-agent
# observations for Simple Spread. Exposing MAPDN's get_state() here would give
# the MAPDN critic the full 62-dim network state instead, i.e. a different
# critic architecture on the second member of the same task family, under a
# design that holds infrastructure fixed. state() below stays implemented so
# the option remains open, but the task must pass return_state=False.
BENCHMARL_HAS_STATE = False

VALID_BARRIERS = tuple(Voltage_Barrier.keys())

# reset(options=...) keys that select a fixed episode start time
_MANUAL_RESET_KEYS = frozenset({"day", "hour", "interval"})


class MAPDNParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper for MAPDN distributed active voltage control.

    Wraps the PyMARL-style :class:`VoltageControl` without touching its physics.
    Each agent controls one PV inverter and observes only its own zone, which is
    the partial observability the underlying Dec-POMDP is built on.

    Information exposed to agents is deliberately limited to the per-agent
    observation. See :meth:`get_diagnostics` for why ``infos`` is empty.
    """

    metadata = {
        "name": "mapdn_voltage_control",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(self, config: dict[str, Any]):
        super().__init__()

        self.config = deepcopy(config)
        self._validate_config(self.config)

        self.render_mode = None

        self._env = VoltageControl(self.config)

        self.num_mapdn_agents = self._env.n_agents
        self.obs_dim = self._env.get_obs_size()
        self.state_dim = self._env.get_state_size()
        self.action_dim = self._env.get_total_actions()

        # "agent_<i>" is the convention BenchMARL and TorchRL expect; the index
        # is MAPDN's sgen index, so agent_i controls sgen i (see _flatten_actions).
        self.possible_agents = [f"agent_{i}" for i in range(self.num_mapdn_agents)]

        # MAPDN's agent set is fixed by the network config, so the live list is
        # populated up front rather than only after the first reset.
        self.agents = list(self.possible_agents)

        self.observation_spaces = {
            agent: Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # distributed MAPDN has one continuous control value per PV inverter
        self._action_low = np.full(
            (self.action_dim,), self._env.action_space.low, dtype=np.float32
        )
        self._action_high = np.full(
            (self.action_dim,), self._env.action_space.high, dtype=np.float32
        )

        self.action_spaces = {
            agent: Box(
                low=self._action_low,
                high=self._action_high,
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self._last_diagnostics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        """fail at the wrapper boundary rather than deep inside MAPDN

        every value checked here changes the task definition, so a run that
        silently picked up a different one would not be comparable with the rest
        of the sweep
        """
        mode = config.get("mode")
        if mode != "distributed":
            raise ValueError(
                f"MAPDNParallelEnv supports only mode='distributed', got {mode!r}. "
                "In 'decentralised' mode an agent controls a whole zone with a "
                "variable number of inverters, which needs a different action space."
            )

        for key in ("action_scale", "action_bias"):
            value = config.get(key)
            if value is None:
                raise ValueError(
                    f"{key!r} must be set explicitly (it ships as null in "
                    "args/env_args/var_voltage_control.yaml). It defines the action "
                    "range, so leaving it unset would make runs incomparable."
                )
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{key!r} must be a number, got {value!r}")

        barrier = config.get("voltage_barrier_type")
        if barrier is None:
            raise ValueError(
                "'voltage_barrier_type' must be set explicitly rather than left to "
                f"MAPDN's internal default. Choose one of {VALID_BARRIERS}."
            )
        if barrier not in Voltage_Barrier:
            raise ValueError(
                f"unknown voltage_barrier_type {barrier!r}, expected one of "
                f"{VALID_BARRIERS}"
            )

        history = config.get("history", 1)
        if not isinstance(history, int) or history < 1:
            raise ValueError(f"'history' must be a positive int, got {history!r}")

        episode_limit = config.get("episode_limit")
        if not isinstance(episode_limit, int) or episode_limit < 2:
            raise ValueError(
                f"'episode_limit' must be an int >= 2, got {episode_limit!r}"
            )

    @property
    def task_signature(self) -> dict[str, Any]:
        """the config values that define the task, for recording alongside results

        anything in here differing between two runs makes them incomparable
        """
        return {
            "mode": self.config["mode"],
            "voltage_barrier_type": self.config["voltage_barrier_type"],
            "action_scale": self.config["action_scale"],
            "action_bias": self.config["action_bias"],
            "action_low": float(self._action_low[0]),
            "action_high": float(self._action_high[0]),
            "history": self.config.get("history", 1),
            "episode_limit": self.config["episode_limit"],
            "max_steps": self.max_steps,
            "v_upper": self.config.get("v_upper"),
            "v_lower": self.config.get("v_lower"),
            "voltage_weight": self.config.get("voltage_weight"),
            "q_weight": self.config.get("q_weight"),
            "line_weight": self.config.get("line_weight"),
            "state_space": self.config.get("state_space"),
            "data_path": self.config.get("data_path"),
            "n_agents": self.num_mapdn_agents,
            "obs_dim": self.obs_dim,
            "benchmarl_has_state": BENCHMARL_HAS_STATE,
        }

    @property
    def max_steps(self) -> int:
        """number of transitions in a full episode

        MAPDN starts its counter at 1 and increments before comparing against
        episode_limit, so an episode yields episode_limit - 1 transitions. This
        is the value BenchMARL's max_steps should use.
        """
        return self._env.episode_limit - 1

    @property
    def group_map(self) -> dict[str, list[str]]:
        """all agents in one group named "agents"

        Pass this to TorchRL's PettingZooWrapper(group_map=...). TorchRL would
        otherwise infer the group name from the agent-name prefix, and matching
        VMAS's "agents" keeps model configs shared across the task family.
        """
        return {"agents": list(self.possible_agents)}

    @property
    def agent_topology(self) -> dict[str, dict[str, Any]]:
        """which bus and zone each agent sits on

        Not used by the env itself; kept for analysis that needs to relate agent
        indices back to network structure (e.g. communication saliency against
        electrical distance).
        """
        sgen = self._env.base_powergrid.sgen
        return {
            agent: {
                "sgen_index": i,
                "bus": int(sgen["bus"][i]),
                "zone": str(sgen["name"][i]),
            }
            for i, agent in enumerate(self.possible_agents)
        }

    # ------------------------------------------------------------------
    # spaces
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Box:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """reset the episode

        seed: reseeds this instance only. Because the RNG is per-instance, two
            environments built from the same config and never reseeded produce
            identical episodes - give each replicate its own seed. TorchRL's
            SerialEnv already assigns a distinct seed per sub-environment.

        options: {"day", "hour", "interval"} pins the episode start time for
            evaluation. Note that a fixed start alone is not fully deterministic
            while config["reset_action"] is True, since the initial reactive
            power is drawn at random; pass a seed as well, or set
            reset_action=False, for a repeatable evaluation episode.
        """
        if seed is not None:
            # reseeds this instance's generator only, including the one driving
            # observation noise; no other env instance is affected
            self._env.seed(seed)

        self.agents = list(self.possible_agents)
        self._last_diagnostics = {}

        # MAPDN's reset already calls get_obs() once and returns the result.
        # Reusing it matters when history > 1, because get_obs() appends to the
        # observation history on every call, so calling it again here would
        # stack a duplicated frame instead of the zero-padded one.
        if options and _MANUAL_RESET_KEYS & set(options):
            agents_obs = self._reset_from_options(options)
        else:
            agents_obs, _ = self._env.reset()

        observations = self._observation_dict(agents_obs)
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def _reset_from_options(self, options: dict):
        """support a fixed start time for reproducible evaluation episodes

        options: {"day": int, "hour": int, "interval": int}. MAPDN's manual_reset
        also disables the initial measurement noise, matching how the original
        authors run deterministic evaluation.

        Keys outside that set are ignored rather than rejected, because the
        PettingZoo API test probes reset() with an arbitrary options dict.
        """
        missing = _MANUAL_RESET_KEYS - set(options)
        if missing:
            raise ValueError(
                f"reset options must specify all of 'day', 'hour', 'interval'; "
                f"missing {sorted(missing)}"
            )
        agents_obs, _ = self._env.manual_reset(
            day=options["day"],
            hour=options["hour"],
            interval=options["interval"],
        )
        return agents_obs

    def step(self, actions: dict[str, np.ndarray]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        flat_actions = self._flatten_actions(actions)

        reward, done, info = self._env.step(flat_actions)

        # one get_obs() call per transition, matching MAPDN's own loop
        observations = self._observation_dict(self._env.get_obs())

        rewards = {agent: float(reward) for agent in self.agents}

        # MAPDN uses one "terminated" flag for BOTH:
        #
        #   1. reaching episode_limit
        #   2. an unsolvable power flow
        #
        # PettingZoo distinguishes those cases.
        physical_failure = bool(info.get("destroy", 0.0))

        terminated = bool(done and physical_failure)
        truncated = bool(done and not physical_failure)

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        # Agents receive no environment info.
        #
        # Every key MAPDN puts in `info` is a whole-network aggregate computed
        # across all buses and all inverters (average_voltage, total_line_loss,
        # percentage_of_v_out_of_control, ...). There is no per-agent version of
        # them to hand back. Returning them per agent would give every agent the
        # same global summary, i.e. a free always-on communication channel
        # present even in the no-comm baseline, which would collapse the
        # independent variable this environment exists to isolate.
        #
        # They are still needed to report results, so they are kept out of band
        # in get_diagnostics() where they cannot reach a policy input.
        #
        # Keeping this empty at reset AND step also matters mechanically:
        # TorchRL builds its info spec from the reset infos, so a mismatch
        # between the two is what would silently put these keys into
        # observation_spec.
        infos = {agent: {} for agent in self.agents}

        self._last_diagnostics = {
            key: float(value) for key, value in info.items()
        }

        if done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _flatten_actions(self, actions: dict[str, np.ndarray]) -> np.ndarray:
        """dict keyed by agent -> the ordered vector MAPDN expects

        Order follows possible_agents, whose index is MAPDN's sgen index, so
        agent_i's action lands on sgen i. Missing keys raise rather than shifting
        another agent's action onto the wrong inverter.
        """
        missing = [agent for agent in self.possible_agents if agent not in actions]
        if missing:
            raise KeyError(
                f"step() is missing actions for {missing}; MAPDN needs one action "
                "per inverter and a partial dict would misalign the rest."
            )

        flat = np.concatenate(
            [
                np.asarray(actions[agent], dtype=np.float32).reshape(-1)
                for agent in self.possible_agents
            ],
            axis=0,
        )
        # The env clips too; doing it here as well keeps the declared Box the
        # authority at the API boundary, so an out-of-range action from any
        # comm condition saturates identically rather than reaching the physics.
        return np.clip(flat, self._action_low[0], self._action_high[0])

    def state(self) -> np.ndarray:
        """MAPDN's global state, verbatim

        Not part of any agent's observation. Only reaches a model if the
        BenchMARL task opts in via return_state; see BENCHMARL_HAS_STATE.
        """
        return np.asarray(self._env.get_state(), dtype=np.float32)

    def get_diagnostics(self) -> dict[str, float]:
        """whole-network metrics from the last step, for logging only

        Includes MAPDN's headline controllability metrics
        (percentage_of_v_out_of_control, totally_controllable_ratio) plus
        voltage, line-loss and reactive-power summaries.

        Deliberately not routed through `infos`: these are global quantities and
        must never enter an agent's observation. Read them from an evaluation
        callback, not from the policy path.
        """
        return dict(self._last_diagnostics)

    def render(self):
        raise NotImplementedError(
            "MAPDNParallelEnv does not implement rendering; metadata['render_modes'] "
            "is empty."
        )

    def close(self):
        self._env.close()

    def _observation_dict(self, agents_obs) -> dict[str, np.ndarray]:
        return {
            agent: np.asarray(agents_obs[i], dtype=np.float32)
            for i, agent in enumerate(self.possible_agents)
        }
