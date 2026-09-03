"""
commstudy/tasks/vmas/scenarios/registry.py
 
Maps a custom vmas task name (lowercase, matching the CustomVmasTask enum
member's .name.lower()) to the BaseScenario subclass that implements it.
 
To add a new custom vmas scenario:
    1. Drop a new Scenario file in this scenarios/ package.
    2. Add one line to _SCENARIO_REGISTRY below.
    3. Add ONE_MORE = None to CustomVmasTask in ../tasks.py.
    4. Add configs/task_defaults/vmas/<name>.yaml with the scenario's
       make_world(**kwargs) defaults.
    5. Add configs/tasks/vmas_<name>.yaml (the `params:` override layer),
       and one line to commstudy/tasks/registry.py's _TASK_REGISTRY.
No other file in commstudy/tasks/vmas/ needs to change.
"""


from __future__ import annotations
from typing import Type

from vmas.simulator.scenario import BaseScenario

from commstudy.tasks.vmas.scenarios.predator_capture_prey import PredatorCapturePreyScenario

_SCENARIO_REGISTRY: dict[str, Type[BaseScenario]] = {
    "predator_capture_prey": PredatorCapturePreyScenario,
}

def get_scenario_class(task_name: str) -> Type[BaseScenario]:
    try:
        return _SCENARIO_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(_SCENARIO_REGISTRY))
        raise ValueError(
            f"No scenario registered for custom vmas task '{task_name}'. "
            f"Available: {available}"
        ) from exc
