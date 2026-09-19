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
from commstudy.experiments.config import ExperimentSpec, validate_experiment_spec
from commstudy.experiments.evaluation import EvaluationIsolatedExperiment
from commstudy.experiments.metrics import ExperimentMetricsCallback
from commstudy.experiments.health import TrainingHealthCallback
from commstudy.experiments.returns import RETURN_GROUPS_KEY
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

    values["layer_class"] = import_from_path(layer_class_path)

    values["activation_class"] = import_from_path(activation_class_path)

    values["norm_class"] = None if norm_class_path is None else import_from_path(norm_class_path)

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
    from commstudy.experiments.schema import validated_model_config

    # Validate before selecting a branch, and use the same explicit defaults
    # exported into manifests. No module or random parameter is created here.
    config = validated_model_config(config)
    if "groups" in config:
        return EnsembleModelConfig(
            {
                group: _build_single_model_config(group_config)
                for group, group_config in config["groups"].items()
            }
        )
    return _build_single_model_config(config)


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
        f"Unknown model_type '{model_type}'. Expected 'benchmarl_mlp' or 'communication'."
    )


def _build_experiment_config(
    overrides: Mapping[str, Any],
) -> BenchMARLExperimentConfig:
    """
    Start with BenchMARL's standard ExperimentConfig and apply only
    the experiment-level values specified by this project.
    """
    config = BenchMARLExperimentConfig.get_from_yaml()

    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown BenchMARL experiment configuration field '{key}'.")

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
    validate_experiment_spec(spec)
    algorithm_params = spec.algorithm_config.get(
        "params",
        {},
    )

    task_params = spec.task_config.get(
        "params",
        {},
    )

    algorithm_config = resolve_algorithm(
        spec.algorithm,
        algorithm_params,
    )

    task = resolve_task(
        spec.task,
        task_params,
    )

    model_config = build_model_config(spec.model_config)

    critic_model_config = build_model_config(spec.critic_model)

    experiment_config = _build_experiment_config(spec.experiment)

    # BenchMARL creates only the generated run-name child and deliberately
    # uses ``parents=False``. Creating this parent here keeps every caller of
    # the public build path safe, including unmanaged integration tests.
    if experiment_config.save_folder is not None:
        Path(experiment_config.save_folder).mkdir(
            parents=True,
            exist_ok=True,
        )

    configured_callbacks = list(callbacks or ())
    health_callbacks = [c for c in configured_callbacks if isinstance(c, TrainingHealthCallback)]
    if len(health_callbacks) > 1:
        raise ValueError("Only one TrainingHealthCallback may be installed.")
    effective_callbacks = (health_callbacks or [TrainingHealthCallback()]) + [
        c for c in configured_callbacks if not isinstance(c, TrainingHealthCallback)
    ]
    return EvaluationIsolatedExperiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=spec.seed,
        config=experiment_config,
        callbacks=effective_callbacks,
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
    protocol_binding: Mapping[str, Any] | None = None,
    expected_contract: Mapping[str, Any] | None = None,
    validation_run: bool = False,
) -> Experiment:
    """Execute one run with durable metadata, status, and tidy metrics."""
    validate_experiment_spec(spec)
    launch = None
    if spec.task == "vmas_predator_capture_prey":
        from commstudy.experiments.protocols import ProtocolGateError, validate_protocol_launch

        if validation_run:
            frame_limit = spec.experiment.get("max_n_frames")
            if type(frame_limit) is not int or not 1 <= frame_limit <= 6000:
                raise ProtocolGateError(
                    "Validation runs require an explicit integer max_n_frames in [1, 6000]."
                )
        elif protocol_binding is None:
            raise ProtocolGateError("Managed PCP training requires an approved protocol binding.")
        elif expected_contract is None:
            raise ProtocolGateError(
                "Managed PCP scientific training requires a built model contract."
            )
        else:
            launch = validate_protocol_launch(spec, protocol_binding, repo_root=repo_root)
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

    if launch is not None or validation_run:
        from commstudy.experiments.bookkeeping import atomic_write_json, read_json

        metadata = read_json(context.metadata_path, {})
        metadata["execution_purpose"] = "validation" if validation_run else "scientific"
        metadata["protocol"] = launch
        atomic_write_json(context.metadata_path, metadata)

    try:
        experiment = build_experiment(
            managed_spec,
            callbacks=[
                ExperimentMetricsCallback(
                    context.run_dir,
                    heartbeat=recorder.heartbeat,
                    return_groups=managed_spec.task_config.get(RETURN_GROUPS_KEY),
                ),
                *callbacks,
            ],
        )
        recorder.record_experiment(experiment)
        if expected_contract is not None:
            from commstudy.experiments.protocols import actual_model_contract, ProtocolGateError

            if actual_model_contract(experiment) != expected_contract:
                raise ProtocolGateError(
                    "Built actor/critic/task contract differs from the manifest."
                )
        experiment.run()
        recorder.complete(experiment)
        return experiment
    except BaseException as error:
        recorder.fail(error)
        raise
