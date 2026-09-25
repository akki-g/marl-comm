"""BenchMARL binding for the versioned, split-bound MAPDN task.

Optional power-grid imports occur only when constructing an environment, so
discovering/validating a PCP configuration does not require pandapower.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from benchmarl.environments.common import Task, TaskClass
from omegaconf import OmegaConf
import torch
from torchrl.data import Composite, Unbounded
from torchrl.envs import PettingZooWrapper, TransformedEnv, Transform


from commstudy.tasks.mapdn import TaskConfig


METRICS = (
    "percentage_of_v_out_of_control",
    "percentage_of_lower_than_lower_v",
    "percentage_of_higher_than_upper_v",
    "totally_controllable_ratio",
    "average_voltage_deviation",
    "average_voltage",
    "max_voltage_drop_deviation",
    "max_voltage_rise_deviation",
    "total_line_loss",
    "q_loss",
    "destroy",
    "reward_row",
    "observation_row",
    "action_solver_failure",
    "advance_solver_failure",
    "pv_curtailed_mw",
    "voltage_violation_magnitude",
    "voltage_metrics_valid",
    "voltage_bus_count",
    "voltage_violating_bus_count",
    "feasible_step",
    "reward",
    "valid",
)


class MAPDNMetricsTransform(Transform):
    """Copy every transition's diagnostics before an automatic reset can erase it."""

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def transform_observation_spec(self, observation_spec):
        result = observation_spec.clone()
        n = self.adapter.num_mapdn_agents
        result["agents", "info"] = Composite(
            {
                name: Unbounded(shape=(n, 1), dtype=torch.float32, device=result.device)
                for name in METRICS
            },
            shape=(n,),
            device=result.device,
        )
        return result

    def _write(self, td, values):
        n = self.adapter.num_mapdn_agents
        for name in METRICS:
            td.set(
                ("agents", "info", name),
                torch.full(
                    (n, 1), float(values.get(name, 0.0)), dtype=torch.float32, device=td.device
                ),
            )
        return td

    def _reset(self, tensordict, tensordict_reset):
        return self._write(tensordict_reset, {})

    def _step(self, tensordict, next_tensordict):
        return self._write(next_tensordict, self.adapter.get_diagnostics())


def _make_adapter(config, seed=None, split=None):
    from commstudy.tasks.mapdn_data import load_split_manifest, TrainingNormalizer
    from commstudy.adapters.mapdn_parallel import ResearchMAPDNParallelEnv

    values = asdict(TaskConfig(**config))
    manifest = load_split_manifest(values["manifest_path"], data_path=values["data_path"])
    if manifest["max_steps"] != values["max_steps"] or manifest["history"] != values["history"]:
        raise ValueError("Task horizon/history do not match the data manifest.")
    normalizer_path = values.pop("normalization_path")
    normalizer = TrainingNormalizer.load(normalizer_path) if normalizer_path else None
    if normalizer is not None and normalizer.to_dict().get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Normalizer was fitted against a different data manifest.")
    declared_settings = manifest.get("settings", {})
    actual_settings = {
        key: value for key, value in values.items() if key not in {"data_path", "manifest_path"}
    }
    if (normalizer is not None or declared_settings) and declared_settings != actual_settings:
        raise ValueError(
            "Data manifest settings do not match the task/normalizer calibration; "
            "regenerate a declared manifest and fit on its training split."
        )
    values["split"] = split or values["split"]
    values["seed"] = seed
    values["episode_limit"] = values["max_steps"] + 1
    values["solver_kwargs"] = {
        "algorithm": values.pop("solver_algorithm"),
        "max_iteration": values.pop("solver_max_iteration"),
        "tolerance_mva": values.pop("solver_tolerance_mva"),
        "numba": False,
        "init": "auto",
        "calculate_voltage_angles": True,
    }
    adapter = ResearchMAPDNParallelEnv(values, manifest, normalizer)
    if list(adapter._env.base_powergrid.sgen.index) != list(range(adapter.num_mapdn_agents)):
        raise ValueError("MAPDN inverter indices must be contiguous in declared agent order.")
    return adapter


def make_mapdn_env(config, seed=None, device="cpu", split=None):
    adapter = _make_adapter(config, seed, split)
    env = PettingZooWrapper(
        adapter,
        group_map=adapter.group_map,
        return_state=False,
        categorical_actions=False,
        device=device,
    )
    return TransformedEnv(env, MAPDNMetricsTransform(adapter))


class PowerGridTaskClass(TaskClass):
    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        del num_envs
        if not continuous_actions:
            raise ValueError("MAPDN requires continuous inverter actions.")
        config = deepcopy(self.config)
        return lambda: make_mapdn_env(config, seed, device)

    def configure_evaluation_env(self, env):
        current = env
        while isinstance(current, TransformedEnv):
            current = current.base_env
        current._env._env.set_split(self.config["evaluation_split"])

    def supports_continuous_actions(self):
        return True

    def supports_discrete_actions(self):
        return False

    def max_steps(self, env):
        return self.config["max_steps"]

    def has_render(self, env):
        return False

    def group_map(self, env):
        n = env.full_action_spec_unbatched["agents"].shape[-1]
        return {"agents": [f"agent_{i}" for i in range(n)]}

    def _select(self, env, leaf):
        source = env.full_observation_spec_unbatched
        result = source.empty()
        result["agents"] = source["agents"].empty()
        result["agents", leaf] = source["agents", leaf].clone()
        return result

    def observation_spec(self, env):
        return self._select(env, "observation")

    def info_spec(self, env):
        return self._select(env, "info")

    def state_spec(self, env):
        return None

    def action_spec(self, env):
        return env.full_action_spec_unbatched.clone()

    def action_mask_spec(self, env):
        return None

    @staticmethod
    def env_name():
        return "mapdn"

    @staticmethod
    def log_info(batch):
        result = {}
        # Each team measurement is repeated across agents for transport only.
        # Select one copy to count actual environment transitions, including
        # terminal transitions; never average episodes or sum agent copies.
        valid = batch.get(("next", "agents", "info", "valid"))[..., 0, 0] > 0
        count = int(valid.sum())
        result["mapdn/transitions"] = float(count)
        physical_valid = valid & (
            batch.get(("next", "agents", "info", "voltage_metrics_valid"))[..., 0, 0] > 0
        )
        result["mapdn/voltage_observed_transitions"] = float(physical_valid.sum())
        for metric_name, output in (
            ("voltage_bus_count", "bus_transition_count"),
            ("voltage_violating_bus_count", "violated_bus_transition_count"),
        ):
            result[f"mapdn/{output}"] = float(
                batch.get(("next", "agents", "info", metric_name))[..., 0, 0][physical_valid].sum()
            )
        nonphysical = {
            "destroy",
            "action_solver_failure",
            "advance_solver_failure",
            "pv_curtailed_mw",
            "feasible_step",
            "reward",
            "voltage_metrics_valid",
        }
        for name in METRICS:
            if name in {"valid", "reward_row", "observation_row"}:
                continue
            selected = valid if name in nonphysical else physical_valid
            values = batch.get(("next", "agents", "info", name))[..., 0, 0][selected]
            if values.numel():
                result[f"mapdn/{name}"] = float(values.mean())
        result["mapdn/solver_failure_count"] = float(
            batch.get(("next", "agents", "info", "destroy"))[..., 0, 0][valid].sum()
        )
        return result

    def runtime_contract(self, env):
        from commstudy.tasks.mapdn_data import load_split_manifest, TrainingNormalizer
        import hashlib
        import json
        from importlib.metadata import version
        from commstudy.environments import mapdn

        manifest = load_split_manifest(self.config["manifest_path"])
        path = self.config["normalization_path"]
        normalizer = TrainingNormalizer.load(path) if path else None
        source_root = Path(mapdn.__file__).resolve().parent
        origin_path = source_root / "ORIGIN.json"
        origin = json.loads(origin_path.read_text()) if origin_path.exists() else None
        source = hashlib.sha256()
        for source_path in sorted(source_root.rglob("*.py")):
            source.update(source_path.relative_to(source_root).as_posix().encode())
            source.update(b"\0")
            source.update(source_path.read_bytes())
            source.update(b"\0")
        return {
            "task": "mapdn_voltage_control",
            "dynamics": "mapdn_aligned_transition_v1",
            "manifest_sha256": manifest["sha256"],
            "mapdn_source_sha256": source.hexdigest(),
            "mapdn_origin": origin,
            "powergrid_versions": {
                name: version(name)
                for name in ("pandapower", "numpy", "pandas", "gymnasium", "pettingzoo")
            },
            "normalization_sha256": normalizer.sha256 if normalizer else None,
            "actor_observation_keys": ["observation"],
            "critic_information": "concatenated zone-local observations",
            "diagnostics_excluded_from_models": True,
            "history_initialization": "zero_padded",
            "measurement_noise": self.config["measurement_noise"],
            "training_split": self.config["split"],
            "evaluation_split": self.config["evaluation_split"],
            "max_steps": self.config["max_steps"],
            "capacity": "1.2 * training block maximum PV; excess active power curtailed",
            "transition_alignment": "action/reward same row; solved next-row observation",
        }


class PowerGridTask(Task):
    VOLTAGE_CONTROL = None

    @staticmethod
    def associated_class():
        return PowerGridTaskClass

    def get_from_yaml(self, path=None):
        config = (
            None if path is None else OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        )
        return self.get_task(config)

    def get_task(self, config=None):
        values = asdict(TaskConfig(**(config or {})))
        return self.associated_class()(name=self.name, config=values)
