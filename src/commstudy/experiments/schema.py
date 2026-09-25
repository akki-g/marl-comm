"""Validate the complete experiment before creating environments or output files.

Schemas derive field names and types from the pinned BenchMARL dataclasses and
the selected communication constructor. No model is instantiated, so validation
and hashing leave RNG state and the Identity reference path untouched.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import asdict, fields
from typing import get_type_hints

from benchmarl.experiment import ExperimentConfig
from benchmarl.models import MlpConfig

from commstudy.communication.base import CommModule
from commstudy.communication.channel import (
    DropoutChannel,
    GaussianNoiseChannel,
    IdentityChannel,
    QuantizedChannel,
)
from commstudy.models import CommPolicyConfig
from commstudy.utils.imports import import_from_path
from commstudy.utils.validation import (
    dataclass_values,
    finite_tree,
    known_keys,
    mapping,
    number,
    typed,
)


def _signature(cls):
    signature = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)
    return {
        name: (parameter.default, hints.get(name))
        for name, parameter in signature.parameters.items()
        if name not in {"self", "hidden_dim"}
        and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    }


def _constructor_values(cls, values, path, *, base=None):
    options = {**(_signature(base) if base else {}), **_signature(cls)}
    known_keys(values, set(options), path)
    for key, value in values.items():
        annotation = options[key][1]
        if annotation is not None:
            typed(value, annotation, f"{path}.{key}")
    result = {
        key: default
        for key, (default, _) in options.items()
        if default is not inspect.Parameter.empty
    }
    result.update(values)
    missing = set(options) - set(result)
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(sorted(missing))}")
    return result


def validated_channel(config, path="channel"):
    if config is None:
        return {"type": "identity", "mode": "always"}
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        if not config:
            raise ValueError(f"{path} must contain at least one channel.")
        return {"type": "sequential", "channels": [validated_channel(v, path) for v in config]}
    mapping(config, path)
    kind = config.get("type", "identity")
    if kind in {"sequential", "sequence"}:
        known_keys(config, {"type", "channels"}, path)
        children = config.get("channels")
        if not isinstance(children, (list, tuple)):
            raise ValueError(f"{path}.channels must be a nonempty sequence.")
        return validated_channel(children, path)
    classes = {
        "identity": IdentityChannel,
        "dropout": DropoutChannel,
        "gaussian": GaussianNoiseChannel,
        "gaussian_noise": GaussianNoiseChannel,
        "quantized": QuantizedChannel,
        "quantization": QuantizedChannel,
    }
    if kind not in classes:
        raise ValueError(f"{path}: unknown channel type {kind!r}.")
    values = _constructor_values(
        classes[kind], {k: v for k, v in config.items() if k != "type"}, path
    )
    if "p" in values:
        number(values["p"], f"{path}.p", maximum=1)
    if "std" in values:
        number(values["std"], f"{path}.std")
        if values["std"] > 0 and values["mode"] in {"always", "training"}:
            raise ValueError(f"{path}: training Gaussian noise is not replay-safe.")
    if "levels" in values:
        number(values["levels"], f"{path}.levels", minimum=2)
        number(values["clip_value"], f"{path}.clip_value", strict=True)
    canonical = {"gaussian_noise": "gaussian", "quantization": "quantized"}.get(kind, kind)
    return {"type": canonical, **values}


def _communication_params(params, path):
    dataclass_values(CommPolicyConfig, params, path)
    values = {**asdict(CommPolicyConfig()), **params}
    for key in ("hidden_dim", "num_encoder_layers", "num_roles"):
        number(values[key], f"{path}.{key}", minimum=1)
    actor_keys = values["actor_observation_keys"]
    if not actor_keys:
        raise ValueError(f"{path}.actor_observation_keys must not be empty.")
    for key in actor_keys:
        if not (isinstance(key, str) and key) and not (
            isinstance(key, (list, tuple))
            and key
            and all(isinstance(part, str) and part for part in key)
        ):
            raise ValueError(f"{path}.actor_observation_keys contains an invalid key.")
    context = values["comm_context_keys"]
    known_keys(context, {"mask", "sender_mask", "class_id"}, f"{path}.comm_context_keys")
    for key, value in context.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}.comm_context_keys.{key} must be a nonempty string.")
    cls = import_from_path(values["comm_class_path"])
    if not isinstance(cls, type) or not issubclass(cls, CommModule):
        raise ValueError(f"{path}.comm_class_path must select a CommModule class.")
    comm = _constructor_values(cls, values["comm_kwargs"], f"{path}.comm_kwargs", base=CommModule)
    for key in ("message_dim", "key_dim", "num_heads", "rounds", "num_roles"):
        if key in comm:
            typed(comm[key], int, f"{path}.comm_kwargs.{key}")
            number(comm[key], f"{path}.comm_kwargs.{key}", minimum=1)
    for key in ("temperature", "grad_clip"):
        if comm.get(key) is not None:
            number(comm[key], f"{path}.comm_kwargs.{key}", strict=True)
    for key in ("sender_budget", "negative_slope"):
        if comm.get(key) is not None:
            number(comm[key], f"{path}.comm_kwargs.{key}")
    for key in ("gate_threshold", "erdos_renyi_p"):
        if key in comm:
            number(comm[key], f"{path}.comm_kwargs.{key}", maximum=1)
    if "num_heads" in comm:
        for key in ("message_dim", "key_dim"):
            if comm[key] % comm["num_heads"]:
                raise ValueError(f"{path}.comm_kwargs.{key} must be divisible by num_heads.")
    if "topology" in comm and comm["topology"] not in {"full", "directed_ring", "erdos_renyi"}:
        raise ValueError(f"{path}.comm_kwargs.topology is unknown.")
    if "sender_selection" in comm:
        modes = {"learned", "random"} if cls.__name__ == "GatedComm" else {"attention", "random"}
        if comm["sender_selection"] not in modes:
            raise ValueError(f"{path}.comm_kwargs.sender_selection must be one of {sorted(modes)}.")
    if "channel" in comm:
        comm["channel"] = validated_channel(comm["channel"], f"{path}.comm_kwargs.channel")
    if cls.__name__ == "IdentityComm" and (
        comm.get("normalize_comm_path") or comm.get("grad_clip") is not None
    ):
        raise ValueError(f"{path}: IdentityComm must remain parameterless and have no path guards.")
    values["comm_kwargs"] = comm
    return values


def validated_model_config(config, path="model_config"):
    """Return a defaults-complete model document after rejecting ignored paths."""
    mapping(config, path)
    if "groups" in config:
        known_keys(config, {"groups"}, path)
        groups = mapping(config["groups"], f"{path}.groups")
        if not groups:
            raise ValueError(f"{path}.groups must not be empty.")
        if any("groups" in mapping(value, path) for value in groups.values()):
            raise ValueError(f"{path}: nested grouped models are not supported.")
        return {
            "groups": {
                group: validated_model_config(value, f"{path}.groups.{group}")
                for group, value in groups.items()
            }
        }
    known_keys(config, {"model_type", "params"}, path)
    kind = config.get("model_type")
    params = mapping(config.get("params", {}), f"{path}.params")
    finite_tree(params, f"{path}.params")
    if kind == "communication":
        values = _communication_params(params, f"{path}.params")
    elif kind == "benchmarl_mlp":
        fields_by_name = {field.name for field in fields(MlpConfig)}
        paths = {"layer_class_path", "activation_class_path", "norm_class_path"}
        class_keys = {"layer_class", "activation_class", "norm_class"}
        known_keys(params, (fields_by_name - class_keys) | paths, f"{path}.params")
        values = asdict(MlpConfig.get_from_yaml())
        for key in class_keys:
            cls = values.pop(key)
            values[f"{key}_path"] = None if cls is None else f"{cls.__module__}.{cls.__name__}"
        # Preserve the runner's documented class defaults rather than upstream's activation.
        values.update(
            layer_class_path="torch.nn.Linear",
            activation_class_path="torch.nn.Tanh",
            norm_class_path=None,
        )
        values.update(params)
        hints = get_type_hints(MlpConfig)
        for key, value in params.items():
            if key not in paths:
                typed(value, hints[key], f"{path}.params.{key}")
        if not values["num_cells"]:
            raise ValueError(f"{path}.params.num_cells must not be empty.")
        for cell in values["num_cells"]:
            number(cell, f"{path}.params.num_cells", minimum=1)
        number(values["num_feature_dims"], f"{path}.params.num_feature_dims", minimum=1)
    else:
        raise ValueError(f"{path}: unknown model_type {kind!r}.")
    for key in ("layer_class_path", "activation_class_path", "norm_class_path"):
        value = values.get(key)
        if value is not None:
            typed(value, str, f"{path}.params.{key}")
            import_from_path(value)
    return {"model_type": kind, "params": values}


def validate_experiment_values(values):
    dataclass_values(ExperimentConfig, values, "experiment")
    for key, value in values.items():
        if value is None:
            continue
        if key in {
            "gamma",
            "polyak_tau",
            "exploration_eps_init",
            "exploration_eps_end",
            "off_policy_prb_alpha",
            "off_policy_prb_beta",
        }:
            number(value, f"experiment.{key}", maximum=1)
        elif key in {"lr", "adam_eps", "clip_grad_val"}:
            number(value, f"experiment.{key}", strict=True)
        elif isinstance(value, int) and not isinstance(value, bool):
            zero_allowed = key in {"checkpoint_interval", "off_policy_init_random_frames"}
            number(value, f"experiment.{key}", minimum=0 if zero_allowed else 1)
    finite_tree(values, "experiment")


def validate_spec_documents(spec):
    from commstudy.algorithms import resolve_algorithm
    from commstudy.tasks import resolve_task

    typed(spec.seed, int, "seed")
    number(spec.seed, "seed", maximum=2**32 - 1)
    for key in ("algorithm", "task", "model"):
        typed(getattr(spec, key), str, key)
    known_keys(spec.algorithm_config, {"params"}, "algorithm_config")
    known_keys(spec.task_config, {"params", "return_groups"}, "task_config")
    resolve_algorithm(
        spec.algorithm, mapping(spec.algorithm_config.get("params", {}), "algorithm_config.params")
    )
    resolve_task(spec.task, mapping(spec.task_config.get("params", {}), "task_config.params"))
    groups = {"adversary", "agent"} if spec.task == "vmas_predator_capture_prey" else {"agents"}
    returns = spec.task_config.get("return_groups")
    if returns is not None:
        typed(returns, list[str], "task_config.return_groups")
        if not returns or len(returns) != len(set(returns)) or not set(returns) <= groups:
            raise ValueError("task_config.return_groups must contain distinct known agent groups.")
    for path, config in (("model_config", spec.model_config), ("critic_model", spec.critic_model)):
        validated = validated_model_config(config, path)
        if "groups" in validated:
            if set(validated["groups"]) != groups:
                raise ValueError(f"{path}.groups must match task groups {sorted(groups)}.")
            members = validated["groups"].values()
        else:
            members = [validated]
        for member in members:
            params = member["params"]
            if member["model_type"] == "communication":
                if path == "critic_model":
                    raise ValueError(
                        "Communication models are actor-only; use the centralized MLP."
                    )
                if spec.task == "vmas_predator_capture_prey" and (
                    params["use_role_embedding"] or params["comm_kwargs"].get("role_aware")
                ):
                    raise ValueError(
                        "PCP has no declared predator roles; role conditioning is disabled."
                    )
    validate_experiment_values(spec.experiment)
