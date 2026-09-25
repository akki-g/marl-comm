"""Load one standalone experiment document and build its five-method seed matrix."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from commstudy.utils.validation import known_keys, mapping, typed, number, finite_tree

METHODS = ("identity", "broadcast", "gated", "attention", "graph")
TASK_NAMES = {"pcp": "vmas_predator_capture_prey", "mapdn": "mapdn_voltage_control"}
COMM_CLASSES = {name: f"commstudy.communication.{name}.{name.title()}Comm" for name in METHODS}


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
    raw["algorithm_config"]["params"] = asdict(
        resolve_algorithm(spec.algorithm, spec.algorithm_config.get("params", {}))
    )
    raw["task_config"]["params"] = deepcopy(
        resolve_task(spec.task, spec.task_config.get("params", {})).config
    )
    raw["model_config"] = validated_model_config(spec.model_config)
    raw["critic_model"] = validated_model_config(spec.critic_model, "critic_model")
    raw["experiment"] = {**asdict(ExperimentConfig.get_from_yaml()), **raw["experiment"]}
    return raw


def _block(document, name, defaults):
    values = document.get(name, {})
    known_keys(values, set(defaults), name)
    return {**deepcopy(defaults), **deepcopy(values)}


def _positive_int(value, name):
    typed(value, int, name)
    number(value, name, minimum=1)


def config_from_dict(document: Mapping, *, base_dir: Path) -> dict:
    """Resolve paths relative to the config, never the invoking working directory."""
    keys = {
        "algorithm",
        "task",
        "seeds",
        "frames",
        "output_dir",
        "algorithm_params",
        "task_params",
        "model",
        "training",
        "evaluation",
        "execution",
        "normalization",
    }
    known_keys(document, keys, "config")
    required = {"algorithm", "task", "seeds", "frames", "output_dir"}
    if missing := required - set(document):
        raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
    result = deepcopy(dict(document))
    finite_tree(result, "config")
    algorithm, task = result["algorithm"], result["task"]
    if algorithm not in ("mappo", "qmix") or task not in TASK_NAMES:
        raise ValueError("algorithm must be mappo or qmix; task must be pcp or mapdn")
    if algorithm == "qmix" and task == "mapdn":
        raise ValueError("QMIX requires discrete actions; MAPDN uses continuous inverter actions.")
    seeds = result["seeds"]
    typed(seeds, list[int], "seeds")
    if not 1 <= len(seeds) <= 5 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain one to five distinct integers")
    for seed in seeds:
        number(seed, "seed", maximum=2**32 - 1)
    _positive_int(result["frames"], "frames")
    typed(result["output_dir"], str, "output_dir")
    if not result["output_dir"].strip():
        raise ValueError("output_dir must be nonempty")
    result["output_dir"] = str((base_dir / result["output_dir"]).resolve())
    execution = _block(result, "execution", {"workers": 5, "device": "cpu"})
    _positive_int(execution["workers"], "execution.workers")
    if execution["device"] not in ("cpu", "cuda"):
        raise ValueError("execution.device must be cpu or cuda")
    result["execution"] = execution
    model = _block(
        result,
        "model",
        {
            "hidden_dim": 128,
            "num_encoder_layers": 2,
            "critic_hidden_sizes": [128, 128],
            "message_dim": 32,
            "key_dim": 32,
            "num_heads": 4,
            "rounds": 1,
        },
    )
    for key in set(model) - {"critic_hidden_sizes"}:
        _positive_int(model[key], f"model.{key}")
    typed(model["critic_hidden_sizes"], list[int], "model.critic_hidden_sizes")
    if not model["critic_hidden_sizes"]:
        raise ValueError("model.critic_hidden_sizes must not be empty")
    for size in model["critic_hidden_sizes"]:
        _positive_int(size, "model.critic_hidden_sizes")
    if model["rounds"] > 3 or any(
        model[k] % model["num_heads"] for k in ("message_dim", "key_dim")
    ):
        raise ValueError("rounds must be <=3 and message/key dimensions must divide into heads")
    result["model"] = model
    training_defaults = {
        "gamma": 0.95 if task == "pcp" else 0.99,
        "lr": 5e-5,
        "adam_eps": 1e-6,
        "clip_grad_val": 5.0,
        "batch_frames": 6000 if task == "pcp" else 256,
        "num_envs": 10 if task == "pcp" else 1,
    }
    if algorithm == "mappo":
        training_defaults.update(
            epochs=10 if task == "pcp" else 4, minibatch_size=400 if task == "pcp" else 64
        )
    else:
        training_defaults.update(
            optimizer_steps=100, train_batch_size=128, memory_size=100000, init_random_frames=6000
        )
    training = _block(result, "training", training_defaults)
    for key, value in training.items():
        if key in {"gamma", "lr", "adam_eps", "clip_grad_val"}:
            number(value, f"training.{key}", strict=True, maximum=1 if key == "gamma" else None)
        elif key == "init_random_frames":
            typed(value, int, f"training.{key}")
            number(value, f"training.{key}")
        else:
            _positive_int(value, f"training.{key}")
    batch = training["batch_frames"]
    if result["frames"] % batch or batch % training["num_envs"]:
        raise ValueError(
            "frames must divide into whole batches; batch_frames must divide by num_envs"
        )
    if algorithm == "mappo" and batch % training["minibatch_size"]:
        raise ValueError("training.batch_frames must divide by minibatch_size")
    result["training"] = training
    evaluation = _block(
        result,
        "evaluation",
        {
            "interval": 12000 if task == "pcp" else 8192,
            "episodes": 128 if task == "pcp" else 16,
            "deterministic": True,
            "static": task == "mapdn",
        },
    )
    for key in ("interval", "episodes"):
        _positive_int(evaluation[key], f"evaluation.{key}")
    for key in ("deterministic", "static"):
        typed(evaluation[key], bool, f"evaluation.{key}")
    if evaluation["interval"] % batch:
        raise ValueError("evaluation.interval must be a multiple of training.batch_frames")
    result["evaluation"] = evaluation
    params = dict(mapping(result.get("task_params", {}), "task_params"))
    if task == "pcp":
        from commstudy.tasks.pcp import TaskConfig

        if "normalization" in result:
            raise ValueError("normalization applies only to MAPDN")
        result["task_params"] = asdict(TaskConfig(**params))
    else:
        from commstudy.tasks.mapdn import TaskConfig

        controlled = {"manifest_path", "normalization_path", "split", "evaluation_split"}
        if controlled & set(params):
            raise ValueError("MAPDN splits and generated artifact paths are managed by the runner")
        data_path = params.get("data_path")
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError(
                "MAPDN requires task_params.data_path to a local case33 data directory"
            )
        params["data_path"] = str((base_dir / data_path).resolve())
        result["task_params"] = {
            k: v for k, v in asdict(TaskConfig(**params)).items() if k not in controlled
        }
        normalization = _block(result, "normalization", {"episodes": 16, "seed": 3200000})
        _positive_int(normalization["episodes"], "normalization.episodes")
        typed(normalization["seed"], int, "normalization.seed")
        number(normalization["seed"], "normalization.seed", maximum=2**32 - 1)
        result["normalization"] = normalization
    from commstudy.algorithms import resolve_algorithm

    params = mapping(result.get("algorithm_params", {}), "algorithm_params")
    result["algorithm_params"] = asdict(resolve_algorithm(algorithm, params))
    # Validate actual actor/critic and framework options without building environments.
    for method in METHODS:
        make_spec(result, method, seeds[0])
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    document = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return config_from_dict(document, base_dir=path.parent)


def make_spec(config: Mapping, method: str, seed: int) -> ExperimentSpec:
    """Assemble one policy with task-owned grouping and the shared model shell."""
    if method not in METHODS or seed not in config["seeds"]:
        raise ValueError("Method/seed is not in this experiment")
    m, t, e = config["model"], config["training"], config["evaluation"]
    pcp, on_policy = config["task"] == "pcp", config["algorithm"] == "mappo"
    kwargs = {}
    if method != "identity":
        kwargs.update(
            message_dim=m["message_dim"],
            exclude_self=True,
            residual=True,
            channel={"type": "identity", "mode": "always"},
        )
    if method in {"attention", "graph"}:
        kwargs.update(key_dim=m["key_dim"], num_heads=m["num_heads"], rounds=m["rounds"])
    if method == "graph":
        kwargs["topology"] = "full"
    actor = {
        "model_type": "communication",
        "params": {
            "hidden_dim": m["hidden_dim"],
            "num_encoder_layers": m["num_encoder_layers"],
            "activation_class_path": "torch.nn.Tanh",
            "actor_observation_keys": ["observation"],
            "comm_class_path": COMM_CLASSES[method],
            "comm_kwargs": kwargs,
            "comm_context_keys": {}
            if method == "identity"
            else {"mask": "comm_mask", "sender_mask": "comm_sender_mask"},
        },
    }
    critic = {
        "model_type": "benchmarl_mlp",
        "params": {"num_cells": m["critic_hidden_sizes"], "activation_class_path": "torch.nn.Tanh"},
    }
    if pcp:
        prey = {
            "model_type": "benchmarl_mlp",
            "params": {"num_cells": [8], "activation_class_path": "torch.nn.Tanh"},
        }
        actor = {"groups": {"adversary": actor, "agent": deepcopy(prey)}}
        critic = {"groups": {"adversary": critic, "agent": deepcopy(prey)}}
    task_params = deepcopy(config["task_params"])
    if not pcp:
        data = Path(config["output_dir"]) / "data"
        task_params.update(
            manifest_path=str(data / "split.json"),
            normalization_path=str(data / "normalizer.json"),
            split="train",
            evaluation_split="validation",
        )
    device = config["execution"]["device"]
    training = {
        "sampling_device": device if pcp else "cpu",
        "train_device": device,
        "buffer_device": device,
        "share_policy_params": True,
        "prefer_continuous_actions": on_policy,
        "collect_with_grad": False,
        "parallel_collection": False,
        "max_n_frames": config["frames"],
        "max_n_iters": None,
        "gamma": t["gamma"],
        "lr": t["lr"],
        "adam_eps": t["adam_eps"],
        "clip_grad_norm": True,
        "clip_grad_val": t["clip_grad_val"],
        "evaluation": True,
        "evaluation_interval": e["interval"],
        "evaluation_episodes": e["episodes"],
        "evaluation_deterministic_actions": e["deterministic"],
        "evaluation_static": e["static"],
        "render": False,
        # BenchMARL conditions scheduled evaluation on native logging being enabled.
        "loggers": [],
        "create_json": True,
        "project_name": "commstudy",
        "checkpoint_interval": 0,
        "checkpoint_at_end": False,
        "exclude_buffer_from_checkpoint": True,
    }
    if on_policy:
        training.update(
            on_policy_collected_frames_per_batch=t["batch_frames"],
            on_policy_n_envs_per_worker=t["num_envs"],
            on_policy_n_minibatch_iters=t["epochs"],
            on_policy_minibatch_size=t["minibatch_size"],
        )
    else:
        training.update(
            off_policy_collected_frames_per_batch=t["batch_frames"],
            off_policy_n_envs_per_worker=t["num_envs"],
            off_policy_n_optimizer_steps=t["optimizer_steps"],
            off_policy_train_batch_size=t["train_batch_size"],
            off_policy_memory_size=t["memory_size"],
            off_policy_init_random_frames=t["init_random_frames"],
        )
    return experiment_spec_from_dict(
        dict(
            algorithm=config["algorithm"],
            task=TASK_NAMES[config["task"]],
            model=method,
            seed=seed,
            algorithm_config={"params": dict(config["algorithm_params"])},
            task_config={
                "params": task_params,
                "return_groups": ["adversary" if pcp else "agents"],
            },
            model_config=actor,
            critic_model=critic,
            experiment=training,
        )
    )


def allocation(config: Mapping) -> list[tuple[str, int]]:
    return [(method, seed) for seed in config["seeds"] for method in METHODS]


def config_fingerprint(config: Mapping) -> str:
    """Concurrency may change on restart; numerical settings may not."""
    value = deepcopy(dict(config))
    value["execution"].pop("workers")
    value.pop("output_dir")
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
