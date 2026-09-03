from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmarl.experiment import (
    Experiment,
    ExperimentConfig as BenchMARLExperimentConfig,
)
from benchmarl.experiment.callback import Callback
from benchmarl.models import MlpConfig, EnsembleModelConfig
from benchmarl.models.common import ModelConfig

from commstudy.algorithms import resolve_algorithm
from commstudy.experiments.bookkeeping import RunContext, RunRecorder
from commstudy.experiments.config import ExperimentSpec
from commstudy.experiments.metrics import ExperimentMetricsCallback
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
    
    if "groups" in config:
        return EnsembleModelConfig(
            {
                group: _build_single_model_config(group_config)
                for group, group_config in config["groups"].items()
            }
        )
    return _build_single_model_config(config)

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Expected 'benchmarl_mlp' or 'communication'."
    )

def _build_single_model_config(
    config: Mapping[str, Any],
) -> ModelConfig:
    model_type = config.get("model_type")
    params = config.get("params", {})

    if not isinstance(params, Mapping):
        raise TypeError("Model 'params' must be a mapping.")

    if model_type == "benchmarl_mlp":
        return _build_benchmarl_mlp(params)

    if model_type == "communication":
        return CommPolicyConfig(**dict(params))

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
    callbacks: Sequence[Callback] | None = None,
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

    # BenchMARL creates only the generated run-name child and deliberately
    # uses ``parents=False``. Creating this parent here keeps every caller of
    # the public build path safe, including unmanaged integration tests.
    if experiment_config.save_folder is not None:
        Path(experiment_config.save_folder).mkdir(
            parents=True,
            exist_ok=True,
        )

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=spec.seed,
        config=experiment_config,
        callbacks=list(callbacks or ()),
    )


def run_experiment(
    spec: ExperimentSpec,
    callbacks: Sequence[Callback] | None = None,
) -> Experiment:
    """
    Build and execute an experiment.

    Returning the experiment object is useful for tests and later
    programmatic analysis of training results.
    """
    experiment = build_experiment(
        spec,
        callbacks=callbacks,
    )

    experiment.run()

    return experiment


def run_managed_experiment(
    spec: ExperimentSpec,
    context: RunContext,
    *,
    repo_root: Path,
    overrides: Sequence[str] = (),
    callbacks: Sequence[Callback] = (),
) -> Experiment:
    """Execute one run with durable metadata, status, and tidy metrics."""
    managed_spec = dataclasses.replace(
        spec,
        experiment={
            **spec.experiment,
            "save_folder": str(context.benchmarl_dir.resolve()),
        },
    )
    recorder = RunRecorder(
        context,
        managed_spec,
        repo_root=repo_root,
        overrides=overrides,
    )
    recorder.start()

    try:
        experiment = build_experiment(
            managed_spec,
            callbacks=[
                ExperimentMetricsCallback(context.run_dir, heartbeat=recorder.heartbeat),
                *callbacks,
            ],
        )
        recorder.record_experiment(experiment)
        experiment.run()
        recorder.complete(experiment)
        return experiment
    except BaseException as error:
        recorder.fail(error)
        raise
