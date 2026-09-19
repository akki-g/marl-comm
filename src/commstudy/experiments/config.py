from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import (
    DictConfig,
    OmegaConf,
)

from commstudy.utils.validation import known_keys, mapping


_SELECTION_KEYS = {
    "algorithm",
    "task",
    "model",
    "critic_model",
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

    result = OmegaConf.load(path)
    if not isinstance(result, DictConfig):
        raise ValueError(f"Configuration {path} must be a mapping.")
    return result


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

        configs/experiments/base.yaml (project defaults)
            ↓
        selected algorithm/task/model configs
        (the algorithm's `experiment` block is merged over the
         project's experiment defaults)
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
    known_keys(base, {"algorithm", "task", "model", "critic_model", "seed", "experiment"},
               "base experiment")

    selection_overrides, remaining_overrides = (
        _extract_selection_overrides(overrides)
    )

    selections = {
        "algorithm": base.get("algorithm"),
        "task": base.get("task"),
        "model": base.get("model"),
        "critic_model": base.get("critic_model"),
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
    known_keys(algorithm_document, {"params", "experiment"}, "algorithm document")

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

    critic_model_document = _load_component(
    config_root,
    "critic_models",          # new config_root subfolder
    str(selections["critic_model"]),
)

    # An algorithm YAML carries two distinct things: `params`, which are
    # algorithm-specific and belong in the BenchMARL AlgorithmConfig, and
    # `experiment`, which are general training settings that belong in
    # BenchMARL's ExperimentConfig. They are split here so that
    # ExperimentSpec never mixes them.
    algorithm_experiment = algorithm_document.pop(
        "experiment",
        {},
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
            "critic_model": selections["critic_model"],
        },
        {
            "experiment": algorithm_experiment,
            "algorithm_config": algorithm_document,
            "task_config": task_document,
            "model_config": model_document,
            "critic_model": critic_model_document,
        },
        cli_config,
    )

    raw = _to_plain_dict(merged)

    return experiment_spec_from_dict(raw)


def validate_experiment_spec(spec: ExperimentSpec) -> None:
    """Validate a programmatic spec as strictly as YAML and CLI inputs."""
    from commstudy.experiments.schema import validate_spec_documents

    validate_spec_documents(spec)


def experiment_spec_from_dict(raw: dict[str, Any]) -> ExperimentSpec:
    """Reload an exact saved commstudy snapshot without reading project YAML."""
    expected = {field.name for field in fields(ExperimentSpec)}
    known_keys(raw, expected, "experiment spec")
    missing = expected - set(raw)
    if missing:
        raise ValueError(f"Experiment spec is missing fields: {', '.join(sorted(missing))}")
    for key in expected - {"algorithm", "task", "model", "seed"}:
        mapping(raw[key], key)
    spec = ExperimentSpec(**deepcopy(dict(raw)))
    validate_experiment_spec(spec)
    return spec


def resolved_experiment_dict(spec: ExperimentSpec) -> dict[str, Any]:
    """Export all effective schema/package defaults as a reloadable snapshot.

    This never constructs a model or changes RNG state. The explicit snapshot
    protects future reconstruction from changes to project or package defaults.
    """
    from benchmarl.experiment import ExperimentConfig
    from commstudy.algorithms import resolve_algorithm
    from commstudy.tasks import resolve_task
    from commstudy.experiments.schema import validated_model_config

    validate_experiment_spec(spec)
    raw = asdict(spec)
    raw["algorithm_config"]["params"] = asdict(resolve_algorithm(
        spec.algorithm, spec.algorithm_config.get("params", {})
    ))
    raw["task_config"]["params"] = deepcopy(resolve_task(
        spec.task, spec.task_config.get("params", {})
    ).config)
    raw["model_config"] = validated_model_config(spec.model_config)
    raw["critic_model"] = validated_model_config(spec.critic_model, "critic_model")
    raw["experiment"] = {**asdict(ExperimentConfig.get_from_yaml()), **raw["experiment"]}
    return raw


_RUNTIME_EXPERIMENT_FIELDS = {
    "sampling_device", "train_device", "buffer_device", "loggers", "project_name",
    "wandb_extra_kwargs", "create_json", "save_folder", "restore_file", "restore_map_location",
    "checkpoint_interval", "checkpoint_at_end", "keep_checkpoints_num",
    "exclude_buffer_from_checkpoint", "render",
}


def canonical_scientific_config(
    spec: ExperimentSpec, *, include_seed: bool = True,
) -> dict[str, Any]:
    """Return effective scientific settings; launch hardware is checked separately."""
    raw = resolved_experiment_dict(spec)
    raw["experiment"] = {
        key: value for key, value in raw["experiment"].items()
        if key not in _RUNTIME_EXPERIMENT_FIELDS
    }
    if not include_seed:
        raw.pop("seed")
    return raw


def scientific_config_sha256(spec: ExperimentSpec, *, include_seed: bool = True) -> str:
    payload = json.dumps(canonical_scientific_config(spec, include_seed=include_seed),
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
