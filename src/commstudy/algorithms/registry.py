from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarl.algorithms import (
    MappoConfig,
    QmixConfig,
)

from benchmarl.algorithms.common import AlgorithmConfig
from commstudy.utils.validation import dataclass_values, number


_ALGORITHM_REGISTRY: dict[str, type[AlgorithmConfig]] = {
    "mappo": MappoConfig,
    "qmix": QmixConfig,
}


def available_algorithms() -> tuple[str, ...]:
    """
    Return the names of algorithms exposed by this project.
    """
    return tuple(sorted(_ALGORITHM_REGISTRY))


def resolve_algorithm(
    name: str,
    params: Mapping[str, Any] | None = None,
) -> AlgorithmConfig:
    """
    Resolve a project-level algorithm name to a BenchMARL
    AlgorithmConfig.

    BenchMARL's own YAML defaults are loaded first. Any params
    supplied by our configuration are then applied as overrides.

    This keeps this registry thin: MAPPO and QMIX themselves remain
    completely owned by BenchMARL.
    """
    try:
        config_class = _ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(available_algorithms())

        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available algorithms: {supported}"
        ) from exc

    config = config_class.get_from_yaml()

    if params is None:
        return config

    dataclass_values(config_class, params, "algorithm_config.params")
    for key in ("entropy_coef", "critic_coef"):
        if key in params:
            number(params[key], f"algorithm_config.params.{key}")
    for key in ("clip_epsilon", "lmbda"):
        if key in params:
            number(params[key], f"algorithm_config.params.{key}", maximum=1)
    if "mixing_embed_dim" in params:
        number(params["mixing_embed_dim"], "algorithm_config.params.mixing_embed_dim", minimum=1)
    for key in ("loss_critic_type", "loss_function"):
        if key in params and params[key] not in {"l1", "l2", "smooth_l1"}:
            raise ValueError(f"algorithm_config.params.{key} must be l1, l2, or smooth_l1.")

    for key, value in params.items():
        if not hasattr(config, key):
            raise ValueError(
                f"Algorithm '{name}' does not have a "
                f"configuration field named '{key}'."
            )

        setattr(
            config,
            key,
            value,
        )

    return config
