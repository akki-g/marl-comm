"""Temporal leakage, provenance, and fixed-normalizer tests independent of MAPDN."""

import copy
import csv
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
import hashlib
import json

import numpy as np
import pytest

from commstudy.tasks.mapdn_data import (
    PROFILE_FILES,
    TrainingNormalizer,
    build_split_manifest,
    load_split_manifest,
    row_to_time,
    save_split_manifest,
    verify_split_manifest,
)


def write_profile(path, values, *, timestamps=None, columns=None):
    if timestamps is None:
        timestamps = [datetime(2020, 1, 1) + timedelta(minutes=3 * i) for i in range(len(values))]
    if columns is None:
        columns = [f"unit_{i}" for i in range(values.shape[1])]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", *columns])
        writer.writerows((str(time), *row) for time, row in zip(timestamps, values, strict=True))


@pytest.fixture
def feeder(tmp_path):
    (tmp_path / "model.p").write_bytes(b"opaque model bytes: not unpickled by data reader")
    values = np.arange(120, dtype=np.float64).reshape(60, 2)
    for name in PROFILE_FILES:
        write_profile(tmp_path / name, values)
    return tmp_path


def rehash(artifact):
    content = {key: value for key, value in artifact.items() if key != "sha256"}
    artifact["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_full_window_cannot_cross_split_with_terminal_and_history(feeder):
    manifest = build_split_manifest(feeder, max_steps=5, history=3)
    assert manifest["splits"]["train"]["block"] == [0, 36]
    assert manifest["splits"]["train"]["starts"] == list(range(2, 31))
    assert manifest["splits"]["validation"]["starts"] == list(range(38, 43))
    assert manifest["splits"]["test"]["starts"] == list(range(50, 55))
    used = []
    for split in manifest["splits"].values():
        lo, hi = split["block"]
        rows = set()
        for start in split["starts"]:
            footprint = set(range(start - 2, start + 6))
            assert min(footprint) >= lo and max(footprint) < hi
            rows |= footprint
        used.append(rows)
    assert not (used[0] & used[1] or used[1] & used[2] or used[0] & used[2])


def test_train_statistics_ignore_extreme_heldout_rows(feeder):
    before = build_split_manifest(feeder, 5)
    values = np.arange(120, dtype=np.float64).reshape(60, 2)
    values[36:] = 1e12
    write_profile(feeder / "pv_active.csv", values)
    after = build_split_manifest(feeder, 5)
    assert before["training_statistics"] == after["training_statistics"]
    assert after["training_statistics"]["pv_max"] == [70.0, 71.0]
    np.testing.assert_allclose(after["training_statistics"]["pv_std"], values[:36].std(axis=0))
    assert before["sha256"] != after["sha256"]


def test_explicit_blocks_gaps_and_stride(feeder):
    blocks = {"train": [1, 21], "validation": [25, 40], "test": [45, 60]}
    result = build_split_manifest(feeder, 5, history=2, boundaries=blocks, stride=3)
    assert result["splits"]["train"]["starts"] == [2, 5, 8, 11, 14]
    assert result["training_statistics"]["row_count"] == 20
    assert result["training_statistics"]["profiles"]["pv_active.csv"]["min"] == [2.0, 3.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_steps": True},
        {"max_steps": 5.5},
        {"history": 0},
        {"stride": 0},
        {"fractions": (0.5, 0.2, 0.2)},
        {"fractions": (0.8, 0.1, 0.1)},
        {"boundaries": {"train": [0, 30], "validation": [29, 45], "test": [45, 60]}},
        {"boundaries": {"train": [0, 30], "validation": [30, 45], "test": [45, 61]}},
        {"expected_n_pv": 3},
        {"expected_n_loads": 3},
    ],
)
def test_invalid_config_and_short_blocks_rejected(feeder, kwargs):
    with pytest.raises(ValueError):
        build_split_manifest(feeder, **({"max_steps": 5, "history": 3} | kwargs))


@pytest.mark.parametrize("mutation", ["shift", "duplicate", "gap", "nan", "width", "columns"])
def test_data_alignment_regularity_dimensions_and_finiteness(feeder, mutation):
    values = np.arange(120, dtype=np.float64).reshape(60, 2)
    timestamps = [datetime(2020, 1, 1) + timedelta(minutes=3 * i) for i in range(60)]
    columns = None
    if mutation == "shift":
        timestamps = [time + timedelta(minutes=3) for time in timestamps]
    elif mutation == "duplicate":
        timestamps[5] = timestamps[4]
    elif mutation == "gap":
        timestamps[5] += timedelta(minutes=1)
    elif mutation == "nan":
        values[5, 0] = np.nan
    elif mutation == "width":
        values = values[:, :1]
    else:
        columns = ["unit_1", "unit_0"]
    write_profile(feeder / "load_reactive.csv", values, timestamps=timestamps, columns=columns)
    with pytest.raises(ValueError):
        build_split_manifest(feeder, 5)


def test_missing_model_rejected(feeder):
    (feeder / "model.p").unlink()
    with pytest.raises(ValueError, match="model.p"):
        build_split_manifest(feeder, 5)


def test_manifest_exclusive_artifact_roundtrip_and_data_tamper(feeder):
    manifest = build_split_manifest(
        feeder, 5, source_revision="source123", settings={"noise": False}
    )
    path = feeder / "split.json"
    save_split_manifest(manifest, path)
    assert load_split_manifest(path, feeder) == manifest
    with pytest.raises(FileExistsError):
        save_split_manifest(manifest, path)
    (feeder / "model.p").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="does not match"):
        load_split_manifest(path, feeder)


def test_manifest_mutation_and_false_window_contract_rejected(feeder):
    manifest = build_split_manifest(feeder, 5)
    manifest["splits"]["train"]["starts"].append(35)
    with pytest.raises(ValueError, match="SHA256"):
        save_split_manifest(manifest, feeder / "bad.json")
    rehash(manifest)
    with pytest.raises(ValueError, match="window contract"):
        verify_split_manifest(manifest, feeder)


def test_normalizer_training_only_constant_features_and_immutable_evaluation(feeder):
    manifest = build_split_manifest(feeder, 5)
    observations = np.asarray([[[1, 7], [3, 7]], [[5, 7], [7, 7]]])
    normalizer = TrainingNormalizer.fit(
        observations, split="train", manifest_sha256=manifest["sha256"]
    )
    assert normalizer.sample_count == 4
    np.testing.assert_allclose(normalizer.transform(observations).mean(axis=(0, 1)), [0, 0])
    assert normalizer.scale[1] == 1
    before = normalizer.to_dict()
    assert np.isfinite(normalizer.transform([[1e9, -1e9]])).all()
    assert normalizer.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        normalizer.mean = (99, 99)
    with pytest.raises(ValueError, match="train split"):
        TrainingNormalizer.fit(observations, split="test")
    path = feeder / "normalizer.json"
    normalizer.save(path)
    loaded = TrainingNormalizer.load(path)
    assert loaded == normalizer and loaded.sha256 == normalizer.sha256
    np.testing.assert_array_equal(
        loaded.transform(observations), normalizer.transform(observations)
    )
    with pytest.raises(FileExistsError):
        normalizer.save(path)


@pytest.mark.parametrize("observations", [[], [1, 2], [[float("nan"), 1]], [[1, float("inf")]]])
def test_invalid_normalizer_fit_rejected(observations):
    with pytest.raises(ValueError):
        TrainingNormalizer.fit(observations, split="train")


def test_normalizer_malformed_eval_and_artifact_rejected():
    normalizer = TrainingNormalizer.fit([[1, 4], [2, 4]], split="train")
    for observations in ([1], [[1, 2, 3]], [[np.inf, 2]]):
        with pytest.raises(ValueError):
            normalizer.transform(observations)
    artifact = normalizer.to_dict()
    artifact["mean"][0] = 999
    with pytest.raises(ValueError, match="SHA256"):
        TrainingNormalizer.from_dict(artifact)
    artifact = normalizer.to_dict()
    artifact["fit_split"] = "validation"
    rehash(artifact)
    with pytest.raises(ValueError, match="non-training"):
        TrainingNormalizer.from_dict(artifact)


def test_legacy_row_mapping_is_explicit_about_grid_assumptions(feeder):
    manifest = build_split_manifest(feeder, 5)
    assert row_to_time(29, manifest) == (0, 1, 9)
    with pytest.raises(ValueError, match="outside"):
        row_to_time(60, manifest)
    invalid = copy.deepcopy(manifest)
    invalid["data"]["interval_seconds"] = 7 * 60
    with pytest.raises(ValueError, match="divisor"):
        row_to_time(0, invalid)
    invalid["data"]["first_timestamp"] = "2020-01-01T00:03:00+00:00"
    with pytest.raises(ValueError, match="midnight"):
        row_to_time(0, invalid)
