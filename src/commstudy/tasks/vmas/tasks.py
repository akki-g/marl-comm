"""
Port custom Vmas Environments into benchmarl VmasVlass
"""

from __future__ import annotations 

import copy
from pathlib import Path
from typing import Callable, Optional, Type

from torchrl.envs import EnvBase
from torchrl.envs.libs import VmasEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.utils import DEVICE_TYPING


from commstudy.tasks.vmas.scenarios.registry import get_scenario_class

_DEFAULTS_DIR = (
    Path(__file__).resolve().parents[4] / "configs" / "tasks" / "defaults"
)


class CustomVmasTaskClass(VmasClass):
    """Shared TaskClass for every CustomVmasTask member. Only get_env_fun
    differs from stock VmasClass: instead of resolving self.name.lower()
    as a vmas.scenarios.<name> string, it looks the scenario class up in
    the project's own scenario registry and instantiates it directly."""
 
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        scenario_cls = get_scenario_class(self.name.lower())
        config = copy.deepcopy(self.config)
        return lambda: VmasEnv(
            scenario=scenario_cls(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

class CustomVmasTask(Task):
    """One member per project-defined vmas scenario. All members share
    CustomVmasTaskClass; get_env_fun disambiguates via scenarios/registry.py."""
 
    PREDATOR_CAPTURE_PREY = None
    # NAVIGATION_SWARM = None   <- future scenario, same pattern
 
    @staticmethod
    def associated_class() -> Type[TaskClass]:
        return CustomVmasTaskClass
 
    def get_from_yaml(self, path: Optional[str] = None) -> TaskClass:
        # Task.get_from_yaml()'s default path points inside the installed
        # benchmarl package, which only has yamls for stock scenarios.
        # Fall back to this repo's own defaults dir instead.
        if path is None:
            path = str(_DEFAULTS_DIR / f"{self.name.lower()}.yaml")
        return super().get_from_yaml(path=path)