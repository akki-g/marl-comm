from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from benchmarl.environments.common import TaskClass
from benchmarl.environments import VmasTask

from commstudy.tasks.vmas import CustomVmasTask

_TASK_REGISTRY = {
    "vmas_simple_spread": VmasTask.SIMPLE_SPREAD,
    "vmas_predator_capture_prey": CustomVmasTask.PREDATOR_CAPTURE_PREY
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

    return task_enum.get_task(
        config=task_config,
    )