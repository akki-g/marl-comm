"""Strict, non-coercing validation for serializable research configuration."""

from __future__ import annotations

import math
import types
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


def mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys.")
    return value


def known_keys(value: Any, allowed: set[str], path: str) -> None:
    unknown = set(mapping(value, path)) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown configuration fields: {', '.join(sorted(unknown))}")


def finite_tree(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite, got {value!r}.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")


def typed(value: Any, annotation: Any, path: str) -> None:
    """Validate Python primitives without OmegaConf's string/bool/int coercions."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        finite_tree(value, path)
        return
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                typed(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path} has invalid value {value!r}; expected {annotation}.")
    if origin is Literal:
        valid = value in args
    elif annotation is type(None):
        valid = value is None
    elif annotation is bool:
        valid = type(value) is bool
    elif annotation is int:
        valid = type(value) is int
    elif annotation is float:
        valid = type(value) in (int, float) and math.isfinite(value)
    elif annotation is str:
        valid = isinstance(value, str)
    elif origin in (list, tuple, Sequence) or annotation in (list, tuple):
        valid = isinstance(value, (list, tuple))
        if valid and args:
            for index, item in enumerate(value):
                typed(item, args[0], f"{path}[{index}]")
    elif origin in (dict, Mapping) or annotation is dict:
        valid = isinstance(value, Mapping)
        if valid and args:
            for key, item in value.items():
                typed(key, args[0], path)
                typed(item, args[1], f"{path}.{key}")
    else:
        valid = isinstance(value, annotation) if isinstance(annotation, type) else True
    if not valid:
        raise ValueError(f"{path} has invalid value {value!r}; expected {annotation}.")
    finite_tree(value, path)


def dataclass_values(cls: type, values: Mapping, path: str) -> None:
    known_keys(values, {field.name for field in fields(cls)}, path)
    hints = get_type_hints(cls)
    for key, value in values.items():
        typed(value, hints[key], f"{path}.{key}")


def number(value: Any, path: str, *, minimum=0, maximum=None, strict=False) -> None:
    typed(value, float, path)
    if value < minimum or (strict and value == minimum) or (
        maximum is not None and value > maximum
    ):
        relation = ">" if strict else ">="
        raise ValueError(f"{path} must be {relation} {minimum} and <= {maximum}, got {value}.")
