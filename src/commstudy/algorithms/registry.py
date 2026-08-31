from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarl.algorithms import (
    MappoConfig,
    QmixConfig,
)

from benchmarl.algorithms.common import AlgorithmConfig


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