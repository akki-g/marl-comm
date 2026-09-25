"""Bind PCP to BenchMARL with isolated simulation and separate actor/critic inputs."""

from __future__ import annotations

import copy
from dataclasses import asdict
from collections.abc import Callable

from torchrl.envs import Compose, EnvBase, StepCounter, TransformedEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.utils import DEVICE_TYPING


from commstudy.environments.pcp import PredatorCapturePreyScenario
from commstudy.adapters.randomness import IsolatedVmasEnv
from commstudy.adapters.critic_state import PCPStateTransform, pcp_runtime_contract
from commstudy.tasks.pcp import TaskConfig
from commstudy.utils.validation import dataclass_values
from omegaconf import OmegaConf


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
        scenario_cls = PredatorCapturePreyScenario
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

    @staticmethod
    def associated_class() -> type[TaskClass]:
        return CustomVmasTaskClass

    def get_from_yaml(self, path: str | None = None) -> TaskClass:
        config = (
            None if path is None else OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        )
        return self.get_task(config)

    def get_task(self, config=None) -> TaskClass:
        values = config or {}
        dataclass_values(TaskConfig, values, "task_config.params")
        return self.associated_class()(name=self.name, config=asdict(TaskConfig(**values)))
