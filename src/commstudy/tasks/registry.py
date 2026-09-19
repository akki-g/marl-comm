from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from benchmarl.environments.common import TaskClass
from benchmarl.environments import VmasTask

from commstudy.tasks.vmas import CustomVmasTask
from commstudy.tasks.torchrl.power_grids import PowerGridTask
from benchmarl.environments.vmas.simple_spread import TaskConfig as SpreadTaskConfig
from commstudy.utils.validation import dataclass_values, number
_TASK_REGISTRY = {
    "vmas_simple_spread": VmasTask.SIMPLE_SPREAD,
    "vmas_predator_capture_prey": CustomVmasTask.PREDATOR_CAPTURE_PREY,
    "mapdn_voltage_control": PowerGridTask.VOLTAGE_CONTROL,
}


def available_tasks() -> tuple[str, ...]:
    """
    Return the names of tasks exposed by this project.
    """
    return tuple(sorted(_TASK_REGISTRY))


def resolve_task(
    name: str,
    params: Mapping[str, Any] | None = None,
) -> TaskClass:
    """
    Resolve a project task name to a BenchMARL TaskClass.

    The stock BenchMARL task configuration is loaded first,
    then our task-specific overrides are applied.
    """
    try:
        task_enum = _TASK_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(available_tasks())

        raise ValueError(
            f"Unknown task '{name}'. "
            f"Available tasks: {supported}"
        ) from exc

    task = task_enum.get_from_yaml()

    if params is None:
        return task

    task_config = deepcopy(task.config)
    task_config.update(dict(params))

    if name == "vmas_simple_spread":
        dataclass_values(SpreadTaskConfig, task_config, "task_config.params")
        for key in ("max_steps", "n_agents"):
            number(task_config[key], f"task_config.params.{key}", minimum=1)

    return task_enum.get_task(
        config=task_config,
    )
