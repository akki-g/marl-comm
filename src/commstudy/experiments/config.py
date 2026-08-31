from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omegaconf import (
    DictConfig,
    OmegaConf,
)


_SELECTION_KEYS = {
    "algorithm",
    "task",
    "model",
}


@dataclass(frozen=True)
class ExperimentSpec:
    """
    High-level experiment description understood by commstudy.

    This object contains no training logic. It simply describes
    which components should be assembled by the experiment runner.
    """

    algorithm: str
    task: str
    model: str
    seed: int

    algorithm_config: dict[str, Any]
    task_config: dict[str, Any]
    model_config: dict[str, Any]

    critic_model: dict[str, Any]
    experiment: dict[str, Any]


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    return OmegaConf.load(path)


def _load_component(
    config_root: Path,
    group: str,
    name: str,
) -> DictConfig:
    path = (
        config_root
        / group
        / f"{name}.yaml"
    )

    return _load_yaml(path)


def _extract_selection_overrides(
    overrides: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    """
    Separate component-selection overrides such as:

        algorithm=mappo
        task=vmas_simple_spread
        model=comm_identity

    from ordinary OmegaConf dot overrides such as:

        seed=1
        experiment.max_n_frames=12000
    """
    selections: dict[str, Any] = {}
    remaining: list[str] = []

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}'. "
                "Overrides must have the form key=value."
            )

        key = override.split(
            "=",
            maxsplit=1,
        )[0]

        if key not in _SELECTION_KEYS:
            remaining.append(override)
            continue

        parsed = OmegaConf.from_dotlist(
            [override]
        )

        selections[key] = parsed[key]

    return selections, remaining


def _to_plain_dict(
    config: DictConfig,
) -> dict[str, Any]:
    result = OmegaConf.to_container(
        config,
        resolve=True,
    )

    if not isinstance(result, dict):
        raise TypeError(
            "Expected configuration to resolve to a mapping."
        )

    return result


def load_experiment_spec(
    config_root: Path,
    overrides: Sequence[str] = (),
) -> ExperimentSpec:
    """
    Load the base experiment configuration, selected component
    configurations, and command-line overrides.

    Merge order:

        Bench project base config
            ↓
        selected algorithm experiment settings
            ↓
        selected algorithm/task/model configs
            ↓
        CLI overrides

    CLI overrides therefore always win.
    """
    base_path = (
        config_root
        / "experiments"
        / "base.yaml"
    )

    base = _load_yaml(base_path)

    selection_overrides, remaining_overrides = (
        _extract_selection_overrides(overrides)
    )

    selections = {
        "algorithm": base.get("algorithm"),
        "task": base.get("task"),
        "model": base.get("model"),
    }

    selections.update(
        selection_overrides
    )

    for key, value in selections.items():
        if value is None:
            raise ValueError(
                f"No '{key}' was selected."
            )

    algorithm_document = _load_component(
        config_root,
        "algorithms",
        str(selections["algorithm"]),
    )

    task_document = _load_component(
        config_root,
        "tasks",
        str(selections["task"]),
    )

    model_document = _load_component(
        config_root,
        "models",
        str(selections["model"]),
    )

    algorithm_experiment = (
        algorithm_document.get(
            "experiment",
            {},
        )
    )

    cli_config = OmegaConf.from_dotlist(
        list(remaining_overrides)
    )

    merged = OmegaConf.merge(
        base,
        {
            "algorithm": selections["algorithm"],
            "task": selections["task"],
            "model": selections["model"],
        },
        {
            "experiment": algorithm_experiment,
            "algorithm_config": algorithm_document,
            "task_config": task_document,
            "model_config": model_document,
        },
        cli_config,
    )

    raw = _to_plain_dict(merged)

    return ExperimentSpec(
        algorithm=str(raw["algorithm"]),
        task=str(raw["task"]),
        model=str(raw["model"]),
        seed=int(raw["seed"]),
        algorithm_config=dict(
            raw["algorithm_config"]
        ),
        task_config=dict(
            raw["task_config"]
        ),
        model_config=dict(
            raw["model_config"]
        ),
        critic_model=dict(
            raw["critic_model"]
        ),
        experiment=dict(
            raw["experiment"]
        ),
    )