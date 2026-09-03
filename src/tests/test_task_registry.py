import pytest

from benchmarl.environments import VmasTask
from benchmarl.environments.common import TaskClass

from commstudy.tasks import available_tasks, resolve_task
from commstudy.tasks.vmas import CustomVmasTask


def test_simple_spread_resolves_to_benchmarl_vmas_task():
    task = resolve_task("vmas_simple_spread")

    assert isinstance(task, TaskClass)
    assert isinstance(task, VmasTask.SIMPLE_SPREAD.associated_class())
    assert task.name == VmasTask.SIMPLE_SPREAD.name
    assert task.env_name() == "vmas"


def test_pcp_resolves_to_custom_vmas_task():
    task = resolve_task("vmas_predator_capture_prey")

    assert isinstance(task, TaskClass)
    assert isinstance(
        task,
        CustomVmasTask.PREDATOR_CAPTURE_PREY.associated_class(),
    )
    assert task.name == CustomVmasTask.PREDATOR_CAPTURE_PREY.name
    assert task.env_name() == "vmas"


def test_pcp_landmark_count_uses_the_kwarg_vmas_consumes():
    """VMAS's ``simple_tag.make_world`` reads ``num_landmarks``.

    An unknown kwarg such as ``num_obstacles`` is only warned about by
    ``ScenarioUtils.check_kwargs_consumed``, so a misnamed key would be
    dropped silently instead of failing the run.
    """
    task = resolve_task("vmas_predator_capture_prey")

    assert task.config["num_landmarks"] == 2
    assert "num_obstacles" not in task.config


def test_unknown_task_raises_clearly():
    with pytest.raises(ValueError) as excinfo:
        resolve_task("not_a_task")

    message = str(excinfo.value)

    assert "not_a_task" in message
    assert "vmas_simple_spread" in message


def test_available_tasks():
    assert available_tasks() == (
        "vmas_predator_capture_prey",
        "vmas_simple_spread",
    )


def test_overrides_are_applied_on_top_of_benchmarl_defaults():
    default = resolve_task("vmas_simple_spread")

    task = resolve_task(
        "vmas_simple_spread",
        {"n_agents": 5},
    )

    assert task.config["n_agents"] == 5

    # Keys we did not override keep BenchMARL's stock values.
    assert task.config["max_steps"] == default.config["max_steps"]

    # Overriding must not mutate the default config.
    assert default.config["n_agents"] == 3


def test_resolved_task_still_uses_stock_benchmarl_integration():
    task = resolve_task("vmas_simple_spread")

    assert task.supports_continuous_actions()
    assert task.supports_discrete_actions()
