"""The two environment/task bindings used by commstudy."""

from collections.abc import Mapping
from typing import Any

from benchmarl.environments.common import TaskClass
from commstudy.adapters.pcp import CustomVmasTask
from commstudy.adapters.mapdn import PowerGridTask

_TASKS = {
    "vmas_predator_capture_prey": CustomVmasTask.PREDATOR_CAPTURE_PREY,
    "mapdn_voltage_control": PowerGridTask.VOLTAGE_CONTROL,
}


def available_tasks() -> tuple[str, ...]:
    return tuple(sorted(_TASKS))


def resolve_task(name: str, params: Mapping[str, Any] | None = None) -> TaskClass:
    try:
        task = _TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task {name!r}; available: {available_tasks()}") from exc
    return task.get_task(config=dict(params or {}))
