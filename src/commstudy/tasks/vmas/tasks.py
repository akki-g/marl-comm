"""Bind PCP to BenchMARL with isolated simulation and separate actor/critic inputs."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable

from torchrl.envs import Compose, EnvBase, StepCounter, TransformedEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.utils import DEVICE_TYPING


from commstudy.tasks.vmas.scenarios.registry import get_scenario_class
from commstudy.tasks.vmas.randomness import IsolatedVmasEnv
from commstudy.tasks.vmas.critic_state import PCPStateTransform, pcp_runtime_contract
from commstudy.tasks.vmas.predator_capture_prey import TaskConfig
from commstudy.utils.validation import dataclass_values
from omegaconf import OmegaConf

_DEFAULTS_DIR = Path(__file__).resolve().parents[4] / "configs" / "tasks" / "defaults"


class CustomVmasTaskClass(VmasClass):
    """Project scenario factory and explicit PCP information contracts.

    The current custom registry contains PCP only. Future tasks need their own
    declared state transform before being added to this binding.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        scenario_cls = get_scenario_class(self.name.lower())
        config = copy.deepcopy(self.config)
        max_steps = config.pop("max_steps")

        def make_env():
            scenario = scenario_cls()
            env = IsolatedVmasEnv(
                scenario=scenario,
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=True,
                clamp_actions=True,
                # VMAS combines its horizon with true terminal events; TorchRL
                # would read that combined signal as terminated and suppress
                # value bootstrapping. Keep the horizon in a time-limit transform.
                max_steps=None,
                **config,
            )
            return TransformedEnv(
                env, Compose(PCPStateTransform(scenario), StepCounter(max_steps=max_steps))
            )

        return make_env

    def observation_spec(self, env):
        # Explicit allowlist: privileged state and diagnostic leaves cannot
        # become actor inputs when new wrappers add them to the environment.
        return self._select_spec(env, [(group, "observation") for group in self.group_map(env)])

    def state_spec(self, env):
        return self._select_spec(env, ["state"])

    def runtime_contract(self, env):
        """Semantic task versions and actual actor/critic inputs for provenance."""
        return pcp_runtime_contract(env)

    def info_spec(self, env):
        spec = env.full_observation_spec_unbatched
        keys = [(group, "info") for group in self.group_map(env) if "info" in spec[group]]
        return self._select_spec(env, keys) if keys else None

    @staticmethod
    def _select_spec(env, keys):
        spec = env.full_observation_spec_unbatched
        result = spec.empty()
        for key in keys:
            if isinstance(key, str):
                result[key] = spec[key].clone()
            else:
                group, leaf = key
                # Keep the group composite's agent dimension. Building it by
                # assigning a nested leaf into an empty root loses that shape.
                if group not in result:
                    result[group] = spec[group].empty()
                result[group][leaf] = spec[key].clone()
        return result


class CustomVmasTask(Task):
    """One member per project-defined vmas scenario. All members share
    CustomVmasTaskClass; get_env_fun disambiguates via scenarios/registry.py."""

    PREDATOR_CAPTURE_PREY = None
    # NAVIGATION_SWARM = None   <- future scenario, same pattern

    @staticmethod
    def associated_class() -> type[TaskClass]:
        return CustomVmasTaskClass

    def get_from_yaml(self, path: str | None = None) -> TaskClass:
        # Task.get_from_yaml()'s default path points inside the installed
        # benchmarl package, which only has yamls for stock scenarios.
        # Fall back to this repo's own defaults dir instead.
        if path is None:
            path = str(_DEFAULTS_DIR / f"{self.name.lower()}.yaml")
        config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return self.get_task(config=config)

    def get_task(self, config=None) -> TaskClass:
        if config is None:
            return self.get_from_yaml()
        dataclass_values(TaskConfig, config, "task_config.params")
        return self.associated_class()(name=self.name, config=asdict(TaskConfig(**config)))
