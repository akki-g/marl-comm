from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from benchmarl.experiment import (
    Experiment,
    ExperimentConfig as BenchMARLExperimentConfig,
)
from benchmarl.models import MlpConfig
from benchmarl.models.common import ModelConfig

from commstudy.algorithms import resolve_algorithm
from commstudy.experiments.config import ExperimentSpec
from commstudy.models import CommPolicyConfig
from commstudy.tasks import resolve_task
from commstudy.utils.imports import import_from_path


def _build_benchmarl_mlp(
    params: Mapping[str, Any],
) -> MlpConfig:
    """
    Construct BenchMARL's standard MLP config.

    Our YAML stores classes as import paths so the YAML remains
    serializable. They are resolved here before constructing the
    BenchMARL ModelConfig.
    """
    values = deepcopy(dict(params))

    layer_class_path = values.pop(
        "layer_class_path",
        "torch.nn.Linear",
    )

    activation_class_path = values.pop(
        "activation_class_path",
        "torch.nn.Tanh",
    )

    norm_class_path = values.pop(
        "norm_class_path",
        None,
    )

    values["layer_class"] = import_from_path(
        layer_class_path
    )

    values["activation_class"] = import_from_path(
        activation_class_path
    )

    values["norm_class"] = (
        None
        if norm_class_path is None
        else import_from_path(norm_class_path)
    )

    return MlpConfig(
        **values,
    )


def build_model_config(
    config: Mapping[str, Any],
) -> ModelConfig:
    """
    Construct one of the policy model configurations supported by
    commstudy.

    Currently:

        benchmarl_mlp
            -> BenchMARL MlpConfig

        communication
            -> CommPolicyConfig
    """
    model_type = config.get(
        "model_type"
    )

    params = config.get(
        "params",
        {},
    )

    if not isinstance(params, Mapping):
        raise TypeError(
            "Model 'params' must be a mapping."
        )

    if model_type == "benchmarl_mlp":
        return _build_benchmarl_mlp(
            params
        )

    if model_type == "communication":
        return CommPolicyConfig(
            **dict(params)
        )

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Expected 'benchmarl_mlp' or 'communication'."
    )


def _build_experiment_config(
    overrides: Mapping[str, Any],
) -> BenchMARLExperimentConfig:
    """
    Start with BenchMARL's standard ExperimentConfig and apply only
    the experiment-level values specified by this project.
    """
    config = (
        BenchMARLExperimentConfig.get_from_yaml()
    )

    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(
                "Unknown BenchMARL experiment configuration "
                f"field '{key}'."
            )

        setattr(
            config,
            key,
            value,
        )

    return config


def build_experiment(
    spec: ExperimentSpec,
) -> Experiment:
    """
    Assemble a BenchMARL experiment from a commstudy ExperimentSpec.

    Assembly path:

        ExperimentSpec
            ↓
        task registry
            ↓
        algorithm registry
            ↓
        policy model config
            ↓
        critic model config
            ↓
        BenchMARL Experiment
    """
    algorithm_params = (
        spec.algorithm_config.get(
            "params",
            {},
        )
    )

    task_params = (
        spec.task_config.get(
            "params",
            {},
        )
    )

    algorithm_config = resolve_algorithm(
        spec.algorithm,
        algorithm_params,
    )

    task = resolve_task(
        spec.task,
        task_params,
    )

    model_config = build_model_config(
        spec.model_config
    )

    critic_model_config = build_model_config(
        spec.critic_model
    )

    experiment_config = (
        _build_experiment_config(
            spec.experiment
        )
    )

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=spec.seed,
        config=experiment_config,
    )


def run_experiment(
    spec: ExperimentSpec,
) -> Experiment:
    """
    Build and execute an experiment.

    Returning the experiment object is useful for tests and later
    programmatic analysis of training results.
    """
    experiment = build_experiment(
        spec
    )

    experiment.run()

    return experiment