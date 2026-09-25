"""Explicit MAPDN temporal splits and immutable training observation normalization.

No simulator or Torch imports occur here. CSV statistics describe *unscaled* input
profiles; the adapter must apply its declared demand/PV scales. ``model.p`` is
hashed, never unpickled by this module. Pass feeder dimensions obtained by the
simulator to validate its profile widths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_FILES = ("pv_active.csv", "load_active.csv", "load_reactive.csv")
SPLIT_NAMES = ("train", "validation", "test")
WINDOW_CONVENTION = "[start-history+1,start+max_steps+1)"


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Mapping[str, Any]) -> str:
    content = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(_json_bytes(content)).hexdigest()


def _check_digest(value: Mapping[str, Any]) -> None:
    if value.get("sha256") != _digest(value):
        raise ValueError("artifact SHA256 does not match its contents")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_profiles(data_path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    import pandas as pd

    root = Path(data_path)
    files = {}
    profiles = {}
    reference_times = None
    for name in ("model.p", *PROFILE_FILES):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required nonempty feeder data file missing: {path}")
        files[name] = {"sha256": _file_digest(path), "size_bytes": path.stat().st_size}
        if name == "model.p":
            continue
        frame = pd.read_csv(path)
        if frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ValueError(f"{name} needs at least two rows and one numeric feature")
        try:
            times = pd.to_datetime(frame.iloc[:, 0], errors="raise", format="mixed", utc=True)
            if times.isna().any():
                raise ValueError("missing timestamp")
            # Explicit conversion avoids pandas-version-dependent datetime resolution.
            ticks = times.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid timestamps or numeric values in {name}: {error}") from error
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite numeric values in {name}")
        intervals = np.diff(ticks)
        if intervals[0] <= 0 or not np.all(intervals == intervals[0]):
            raise ValueError(f"{name} timestamps must be increasing and regularly sampled")
        if reference_times is not None and not np.array_equal(reference_times, ticks):
            raise ValueError(f"{name} timestamps do not align with pv_active.csv")
        reference_times = ticks
        profiles[name] = values
        files[name].update(
            rows=len(frame), n_columns=values.shape[1], columns=list(frame.columns[1:])
        )
        if _file_digest(path) != files[name]["sha256"]:
            raise ValueError(f"data changed while reading {name}")
    assert reference_times is not None
    if profiles["load_active.csv"].shape != profiles["load_reactive.csv"].shape:
        raise ValueError("active/reactive load profile dimensions must match")
    if files["load_active.csv"]["columns"] != files["load_reactive.csv"]["columns"]:
        raise ValueError("active/reactive load feature columns must match in order")
    data = {
        "n_rows": len(reference_times),
        "interval_seconds": float((reference_times[1] - reference_times[0]) / 1e9),
        "first_timestamp": pd.Timestamp(reference_times[0], tz="UTC").isoformat(),
        "last_timestamp": pd.Timestamp(reference_times[-1], tz="UTC").isoformat(),
        "files": files,
    }
    return data, profiles


def _blocks(
    n_rows: int,
    fractions: Sequence[float],
    boundaries: Mapping[str, Sequence[int]] | None,
) -> dict[str, list[int]]:
    if boundaries is None:
        if (
            len(fractions) != 3
            or any(isinstance(f, bool) or not math.isfinite(f) or f <= 0 for f in fractions)
            or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("fractions must contain three positive finite values summing to one")
        cut1 = math.floor(n_rows * fractions[0])
        cut2 = math.floor(n_rows * (fractions[0] + fractions[1]))
        boundaries = {"train": (0, cut1), "validation": (cut1, cut2), "test": (cut2, n_rows)}
    if set(boundaries) != set(SPLIT_NAMES):
        raise ValueError("boundaries must define exactly train, validation, and test")
    result = {}
    previous_stop = 0
    for name in SPLIT_NAMES:
        pair = boundaries[name]
        if len(pair) != 2:
            raise ValueError(f"{name} block must be [start, stop)")
        start = _integer(pair[0], f"{name} block start", 0)
        stop = _integer(pair[1], f"{name} block stop", 1)
        if not previous_stop <= start < stop <= n_rows:
            raise ValueError("blocks must be chronological, disjoint, nonempty, and within data")
        result[name] = [start, stop]
        previous_stop = stop
    return result


def _statistics(profiles: Mapping[str, np.ndarray], block: Sequence[int]) -> dict:
    result = {"block": list(block), "row_count": block[1] - block[0], "profiles": {}}
    for name, values in profiles.items():
        train = values[block[0] : block[1]]
        stats = {
            "mean": train.mean(axis=0).tolist(),
            "std": train.std(axis=0, ddof=0).tolist(),
            "min": train.min(axis=0).tolist(),
            "max": train.max(axis=0).tolist(),
        }
        if not all(np.isfinite(v).all() for v in stats.values()):
            raise ValueError(f"nonfinite training statistics for {name}")
        result["profiles"][name] = stats
    result.update(
        pv_std=result["profiles"]["pv_active.csv"]["std"],
        load_active_std=result["profiles"]["load_active.csv"]["std"],
        load_reactive_std=result["profiles"]["load_reactive.csv"]["std"],
        pv_max=result["profiles"]["pv_active.csv"]["max"],
    )
    return result


def build_split_manifest(
    data_path: str | Path,
    max_steps: int,
    history: int = 1,
    *,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    boundaries: Mapping[str, Sequence[int]] | None = None,
    stride: int = 1,
    expected_n_pv: int | None = None,
    expected_n_loads: int | None = None,
    source_revision: str | None = None,
    settings: Mapping[str, Any] | None = None,
) -> dict:
    """Build row-based splits; each complete history/terminal window stays inside one block.

    A reset at row ``r`` reserves ``[r-history+1, r+max_steps+1)``. The
    terminal observation at ``r+max_steps`` is therefore held within the same
    split. Explicit boundaries use absolute row indices and may leave gaps.
    The returned dict is content-addressed; save/load rejects subsequent edits.
    """
    max_steps = _integer(max_steps, "max_steps")
    history = _integer(history, "history")
    stride = _integer(stride, "stride")
    data, profiles = _load_profiles(data_path)
    for expected, name in ((expected_n_pv, "pv_active.csv"), (expected_n_loads, "load_active.csv")):
        if (
            expected is not None
            and _integer(expected, f"{name} expected width") != (profiles[name].shape[1])
        ):
            raise ValueError(f"{name} width does not match the feeder")
    blocks = _blocks(data["n_rows"], fractions, boundaries)
    splits = {}
    for name, (lo, hi) in blocks.items():
        starts = list(range(lo + history - 1, hi - max_steps, stride))
        if not starts:
            raise ValueError(f"{name} block is too short for the complete episode/history window")
        splits[name] = {"block": [lo, hi], "starts": starts}
    result = {
        "schema_version": 1,
        "window_convention": WINDOW_CONVENTION,
        "max_steps": max_steps,
        "history": history,
        "stride": stride,
        "data": data,
        "splits": splits,
        "training_statistics": _statistics(profiles, blocks["train"]),
        "source_revision": source_revision,
        "settings": json.loads(_json_bytes(dict(settings or {}))),
    }
    result["sha256"] = _digest(result)
    return result


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _check_digest(manifest)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("window_convention") != WINDOW_CONVENTION
    ):
        raise ValueError("unsupported split manifest schema or window convention")
    max_steps = _integer(manifest["max_steps"], "max_steps")
    history = _integer(manifest["history"], "history")
    stride = _integer(manifest["stride"], "stride")
    splits = manifest["splits"]
    blocks = _blocks(
        _integer(manifest["data"]["n_rows"], "n_rows"),
        (),
        {name: split["block"] for name, split in splits.items()},
    )
    for name, (lo, hi) in blocks.items():
        starts = splits[name]["starts"]
        if (
            not starts
            or any(isinstance(row, bool) or not isinstance(row, int) for row in starts)
            or starts != list(range(lo + history - 1, hi - max_steps, stride))
        ):
            raise ValueError(f"{name} starts do not match the declared complete-window contract")
    stats = manifest["training_statistics"]
    if stats["block"] != blocks["train"] or stats["row_count"] != (
        blocks["train"][1] - blocks["train"][0]
    ):
        raise ValueError("training statistics must use exactly the training block")


def verify_split_manifest(manifest: Mapping[str, Any], data_path: str | Path) -> None:
    """Recheck hashes, CSV alignment, dimensions, starts, and train-only statistics."""
    _validate_manifest(manifest)
    rebuilt = build_split_manifest(
        data_path,
        manifest["max_steps"],
        manifest["history"],
        boundaries={name: split["block"] for name, split in manifest["splits"].items()},
        stride=manifest["stride"],
        source_revision=manifest["source_revision"],
        settings=manifest["settings"],
    )
    if rebuilt != manifest:
        raise ValueError("split manifest does not match the current feeder data/statistics")


def _save_exclusive(artifact: Mapping[str, Any], path: str | Path) -> None:
    # Exclusive creation makes immutable experiment artifacts non-overwritable.
    encoded = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def save_split_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    _validate_manifest(manifest)
    _save_exclusive(manifest, path)


def load_split_manifest(path: str | Path, data_path: str | Path | None = None) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    _validate_manifest(manifest)
    if data_path is not None:
        verify_split_manifest(manifest, data_path)
    return manifest


def row_to_time(row: int, manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    """Map an absolute row to legacy MAPDN relative (day, hour, interval).

    Absolute-row adapters need not use this compatibility helper. Legacy MAPDN
    assumes a midnight first row and an integral number of samples per hour;
    this function rejects grids violating either assumption.
    """
    row = _integer(row, "row", 0)
    data = manifest["data"]
    if row >= data["n_rows"]:
        raise ValueError("row is outside the feeder dataset")
    first = datetime.fromisoformat(data["first_timestamp"])
    seconds = float(data["interval_seconds"])
    if first.hour or first.minute or first.second or first.microsecond:
        raise ValueError("legacy day/hour mapping requires a midnight dataset start")
    if not math.isfinite(seconds) or seconds < 60 or seconds % 60 or 3600 % seconds:
        raise ValueError("legacy day/hour mapping requires a whole-minute divisor of one hour")
    intervals_per_hour = int(3600 / seconds)
    day, remaining = divmod(row, 24 * intervals_per_hour)
    hour, interval = divmod(remaining, intervals_per_hour)
    return day, hour, interval


@dataclass(frozen=True, slots=True)
class TrainingNormalizer:
    """Fixed featurewise statistics fitted once on explicitly designated training observations.

    The caller controls collection provenance: ``split='train'`` is required,
    and fitting held-out observations is rejected. Evaluation calls only
    ``transform``; no running state or partial-fit method exists. Leading axes
    (for example time and agents) are pooled and the last axis contains features.
    """

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    sample_count: int
    manifest_sha256: str | None = None
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", tuple(float(v) for v in self.mean))
        object.__setattr__(self, "scale", tuple(float(v) for v in self.scale))
        if not self.mean or len(self.mean) != len(self.scale):
            raise ValueError("normalizer mean/scale must have equal nonzero feature widths")
        if (
            not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or any(value <= 0 for value in self.scale)
        ):
            raise ValueError("normalizer statistics must be finite, with positive scales")
        _integer(self.sample_count, "sample_count")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("normalizer epsilon must be positive and finite")
        if self.manifest_sha256 is not None and (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.manifest_sha256)
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA256 hex digest")

    @classmethod
    def fit(
        cls,
        observations: Any,
        *,
        split: str,
        manifest_sha256: str | None = None,
        epsilon: float = 1e-8,
    ) -> TrainingNormalizer:
        if split != "train":
            raise ValueError("observation normalization may only be fitted on the train split")
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim < 2 or not values.size or values.shape[-1] == 0:
            raise ValueError("observations require nonempty sample and feature axes")
        if not np.isfinite(values).all():
            raise ValueError("training observations must be finite")
        values = values.reshape(-1, values.shape[-1])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        scale = np.where(std <= epsilon, 1.0, std)
        return cls(tuple(mean), tuple(scale), len(values), manifest_sha256, epsilon)

    def transform(self, observations: Any) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != len(self.mean):
            raise ValueError("observation feature width does not match the fitted normalizer")
        if not np.isfinite(values).all():
            raise ValueError("observations must be finite")
        result = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        if not np.isfinite(result).all():
            raise ValueError("normalized observations are nonfinite")
        return result

    def to_dict(self) -> dict:
        result = {
            "schema_version": 1,
            "fit_split": "train",
            "mean": list(self.mean),
            "scale": list(self.scale),
            "sample_count": self.sample_count,
            "manifest_sha256": self.manifest_sha256,
            "epsilon": self.epsilon,
        }
        result["sha256"] = _digest(result)
        return result

    @property
    def sha256(self) -> str:
        return self.to_dict()["sha256"]

    @classmethod
    def from_dict(cls, artifact: Mapping[str, Any]) -> TrainingNormalizer:
        _check_digest(artifact)
        if artifact.get("schema_version") != 1 or artifact.get("fit_split") != "train":
            raise ValueError("unsupported normalizer schema or non-training provenance")
        return cls(
            tuple(artifact["mean"]),
            tuple(artifact["scale"]),
            artifact["sample_count"],
            artifact["manifest_sha256"],
            artifact["epsilon"],
        )

    def save(self, path: str | Path) -> None:
        _save_exclusive(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> TrainingNormalizer:
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
