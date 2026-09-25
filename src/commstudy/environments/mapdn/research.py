"""Versioned, split-bound MAPDN dynamics for the commstudy integration.

The earlier adapter/environment remain intact. This explicitly changes temporal
alignment: observations, action solves and rewards share a data row; the next
observation is solved again after advancing exogenous inputs. History is zero
padded at reset. All reset/step noise uses training-fitted scales and one private
environment RNG; action selection mode never controls measurement noise.
"""

from copy import deepcopy

import numpy as np
import pandapower as pp

from .voltage_control.voltage_control_env import VoltageControl


class ResetPowerFlowError(RuntimeError):
    """A declared reset could not be solved; never silently replace its row."""


class ResearchVoltageControl(VoltageControl):
    VERSION = "mapdn_aligned_transition_v1"

    def __init__(self, config, manifest):
        self.manifest = deepcopy(manifest)
        self.split = config["split"]
        self.measurement_noise = config["measurement_noise"]
        self.solver_kwargs = deepcopy(config["solver_kwargs"])
        self.max_steps = config["max_steps"]
        self._fixed_start = None
        self._episode_number = 0
        self._diagnostics = {}
        super().__init__(config)

    def _set_reactive_power_boundary(self):
        stats = self.manifest["training_statistics"]
        self.factor = 1.2
        self.p_max = np.asarray(stats["pv_max"], dtype=np.float64) * self.args.pv_scale
        self.s_max = self.factor * self.p_max
        if np.any(self.s_max <= 0):
            raise ValueError("Every inverter needs positive training-derived capacity.")
        self.pv_std = np.asarray(stats["pv_std"]) * self.args.pv_scale / 100.0
        self.active_demand_std = (
            np.asarray(stats["load_active_std"]) * self.args.demand_scale / 100.0
        )
        self.reactive_demand_std = (
            np.asarray(stats["load_reactive_std"]) * self.args.demand_scale / 100.0
        )

    def _load_network(self):
        network = super()._load_network()
        files = self.manifest["data"]["files"]
        if len(network.sgen) != files["pv_active.csv"]["n_columns"] or (
            len(network.load) != files["load_active.csv"]["n_columns"]
        ):
            raise ValueError("Feeder inverter/load dimensions do not match profile columns.")
        if list(network.sgen.index) != list(range(len(network.sgen))):
            raise ValueError("MAPDN inverter indices must be contiguous in agent order.")
        return network

    def _start_rows(self):
        return self.manifest["splits"][self.split]["starts"]

    def set_split(self, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown data split {split!r}")
        self.split = split
        self._fixed_start = None

    def set_episode_start(self, row):
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError("Episode start must be an integer data row.")
        if int(row) not in self._start_rows():
            raise ValueError("Episode start is outside the declared split/history window.")
        self._fixed_start = int(row)

    def _apply_row(self, row):
        values = (
            self.pv_data.iloc[row].to_numpy(copy=True),
            self.active_demand_data.iloc[row].to_numpy(copy=True),
            self.reactive_demand_data.iloc[row].to_numpy(copy=True),
        )
        if self.measurement_noise:
            scales = (self.pv_std, self.active_demand_std, self.reactive_demand_std)
            values = tuple(x + s * np.abs(self._rng.standard_normal(x.shape))
                           for x, s in zip(values, scales, strict=True))
        pv, active, reactive = values
        # A held-out PV excursion above the training-derived rating is declared
        # active-power curtailment, not sqrt of a negative reactive capability.
        self._curtailed_pv = float(np.maximum(pv - self.s_max, 0).sum())
        self.powergrid.sgen["p_mw"] = np.minimum(pv, self.s_max)
        self.powergrid.load["p_mw"] = active
        self.powergrid.load["q_mvar"] = reactive

    def _solve(self):
        try:
            pp.runpp(self.powergrid, **self.solver_kwargs)
            return bool(self.powergrid.converged) and np.isfinite(
                self.powergrid.res_bus["vm_pu"].to_numpy()).all()
        except pp.ppException:
            return False

    def reset(self, reset_time=True):
        del reset_time
        self.steps = 0
        self.sum_rewards = 0.0
        self.obs_history = {i: [] for i in range(self.n_agents)}
        self._diagnostics = {}
        rows = self._start_rows()
        if not rows:
            raise ValueError("The selected split has no complete episode windows.")
        self._row = (self._fixed_start if self._fixed_start is not None
                     else int(self._rng.choice(rows)))
        self._episode_start_row = self._row
        self._episode_number += 1
        self.powergrid = deepcopy(self.base_powergrid)
        self._apply_row(self._row)
        actions = self.get_action() if self.args.reset_action else np.zeros(self.n_agents)
        self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(
            actions, self.powergrid.sgen["p_mw"].to_numpy())
        if not self._solve():
            raise ResetPowerFlowError(f"Power-flow reset failed at {self.split} row {self._row}.")
        return self.get_obs(), self.get_state()

    def manual_reset(self, day, hour, interval):
        raise ValueError(
            "Research MAPDN uses options={'start_row': integer}; legacy calendar "
            "reset assumes midnight data origins and can change initialization noise."
        )

    def _clip_reactive_power(self, reactive_actions, active_power):
        capacity = np.sqrt(np.maximum(self.s_max ** 2 - np.asarray(active_power) ** 2, 0))
        return capacity * np.asarray(reactive_actions)

    def step(self, actions, add_noise=None):
        if add_noise is not None and add_noise != self.measurement_noise:
            raise ValueError("Measurement noise is immutable within a task protocol.")
        action = np.asarray(actions, dtype=np.float64)
        if action.shape != (self.n_agents,) or not np.isfinite(action).all():
            raise ValueError("Actions must contain one finite scalar per inverter.")
        low, high = self.action_space.low, self.action_space.high
        action = np.clip(action, low, high)
        previous = deepcopy(self.powergrid)
        reward_row = self._row
        curtailed = self._curtailed_pv
        self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(
            action, self.powergrid.sgen["p_mw"].to_numpy())
        action_ok = self._solve()
        if not action_ok:
            self.powergrid = previous
        reward, info = self._calc_reward()
        voltage = self._get_voltage()
        info.update({"reward_row": float(reward_row), "observation_row": float(reward_row),
                     "action_solver_failure": float(not action_ok),
                     "advance_solver_failure": 0.0, "pv_curtailed_mw": curtailed,
                     "voltage_metrics_valid": float(action_ok),
                     "voltage_bus_count": float(len(voltage)),
                     "voltage_violating_bus_count": float(np.count_nonzero(
                         (voltage < self.v_lower) | (voltage > self.v_upper))),
                     "voltage_violation_magnitude": float(np.maximum(
                         self.v_lower - voltage, 0).mean() + np.maximum(
                         voltage - self.v_upper, 0).mean()),
                     "valid": 1.0})
        advance_ok = True
        if action_ok:
            solved = deepcopy(self.powergrid)
            self._row += 1
            self._apply_row(self._row)
            capacity = np.sqrt(np.maximum(self.s_max ** 2 -
                self.powergrid.sgen["p_mw"].to_numpy() ** 2, 0))
            self.powergrid.sgen["q_mvar"] = np.clip(
                self.powergrid.sgen["q_mvar"].to_numpy(), -capacity, capacity)
            advance_ok = self._solve()
            if not advance_ok:
                self.powergrid = solved
                self._row = reward_row
            info["advance_solver_failure"] = float(not advance_ok)
            info["observation_row"] = float(self._row)
        failed = not action_ok or not advance_ok
        info["destroy"] = float(failed)
        info["feasible_step"] = float(not failed)
        if failed:
            reward -= 200.0
            info["totally_controllable_ratio"] = 0.0
        self.steps += 1
        self.sum_rewards += reward
        info["reward"] = float(reward)
        self._diagnostics = info
        return float(reward), bool(failed or self.steps >= self.max_steps), info


