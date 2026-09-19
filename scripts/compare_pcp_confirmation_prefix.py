"""Verify that the replay-check repair leaves the observed training prefix unchanged."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from commstudy.analysis.diagnostics import rebuild_spec
from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.config import resolved_experiment_dict

REPLAY_METRICS = {
    "replay_log_prob_max_abs_error",
    "replay_log_prob_scalar_count",
    "replay_transition_count",
    "replay_batched_log_prob_max_abs_error",
    "replay_batched_log_prob_max_tolerance_ratio",
    "replay_accepted_log_prob_max_tolerance_ratio",
    "replay_collection_shape_fallback_count",
    "replay_collection_shape_fallback_slices",
}
PREFIX_FRAMES = 558000
FIRST_ALLOWED_FALLBACK_FRAME = 564000
CHECKPOINT_FRAMES = 480000
FALLBACK_METRICS = {
    "replay_collection_shape_fallback_count", "replay_collection_shape_fallback_slices",
}
METRIC_COLUMNS = {"timestamp", "frames", "iteration", "phase", "group", "metric", "sample", "value"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs_by_seed(suite):
    runs = {}
    for path in sorted(suite.glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        seed = metadata["seed"]
        if seed in runs:
            raise ValueError(f"Duplicate training seed {seed} in {suite}.")
        runs[seed] = path.parent, metadata
    if set(runs) != {10, 11}:
        raise ValueError(f"Require exactly original confirmation seeds 10 and 11: {suite}.")
    return runs


def canonical_metrics(path, through):
    if type(through) is not int or through != PREFIX_FRAMES:
        raise ValueError(f"This protocol-bound comparison requires exactly {PREFIX_FRAMES} frames.")
    selected, excluded, seen, fallbacks = {}, {}, set(), {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != METRIC_COLUMNS or len(reader.fieldnames) != len(
            METRIC_COLUMNS
        ):
            raise ValueError(f"Unspecified or missing metric CSV columns in {path}.")
        for row in reader:
            frame, iteration = int(row["frames"]), int(row["iteration"])
            if frame < 0 or iteration < 0:
                raise ValueError(f"Negative metric frame/iteration in {path}.")
            metric = row["metric"]
            # Fallback timing is checked over the complete repaired run, even
            # though numeric equality is required only through the fixed prefix.
            if frame > through and metric not in FALLBACK_METRICS:
                continue
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite metric in {path}: {metric}.")
            key = (str(frame), str(iteration), row["phase"], row["group"], metric, row["sample"])
            if key in seen:
                raise ValueError(f"Duplicate prefix/fallback metric in {path}: {key}.")
            seen.add(key)
            if metric.startswith("replay_") and metric not in REPLAY_METRICS:
                raise ValueError(f"Unspecified replay-metric exclusion: {metric}.")
            if metric in FALLBACK_METRICS:
                context = (frame, iteration, row["phase"], row["group"], row["sample"])
                fallbacks.setdefault(context, {})[metric] = value
                if frame < FIRST_ALLOWED_FALLBACK_FRAME and value != 0:
                    raise ValueError(
                        f"Collection-shape fallback occurred before "
                        f"{FIRST_ALLOWED_FALLBACK_FRAME} frames: {path}."
                    )
            if frame > through:
                continue
            if metric == "wall_time_seconds" or metric in REPLAY_METRICS:
                excluded[metric] = excluded.get(metric, 0) + 1
                continue
            selected[key] = value.hex()
    for context, values in fallbacks.items():
        count = values.get("replay_collection_shape_fallback_count")
        slices = values.get("replay_collection_shape_fallback_slices")
        if (
            set(values) != FALLBACK_METRICS or count not in (0, 1)
            or slices < 0 or slices != int(slices) or bool(count) != bool(slices)
        ):
            raise ValueError(f"Inconsistent fallback count/slices in {path}: {context}.")
    if not selected or max(int(key[0]) for key in selected) != through:
        raise ValueError(f"Prefix does not reach exactly {through} recorded frames: {path}.")
    return selected, excluded


def retained_hashes(values):
    return {
        "keys_sha256": hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest(),
        "values_sha256": hashlib.sha256(json.dumps(sorted(values.items())).encode()).hexdigest(),
    }


def scientific_spec(run):
    spec = resolved_experiment_dict(rebuild_spec(run))
    spec["experiment"].pop("save_folder", None)
    return spec


def checkpoint_parameters(original, repaired):
    evidence = {}
    states = []
    for label, run in (("original", original), ("repaired", repaired)):
        paths = list(run.glob("benchmarl/*/checkpoints/checkpoint_480000.pt"))
        if len(paths) != 1:
            raise ValueError(f"Require exactly one preserved 480k checkpoint in {run}.")
        path = paths[0]
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping) or not isinstance(state.get("state"), Mapping):
            raise ValueError(f"Native checkpoint state/header is missing in {path}.")
        header = state["state"]
        if (
            type(header.get("total_frames")) is not int
            or header["total_frames"] != CHECKPOINT_FRAMES
            or type(header.get("n_iters_performed")) is not int
            or header["n_iters_performed"] != 80
        ):
            raise ValueError(f"Native checkpoint header is not the 480k/80-update state: {path}.")
        if {key for key in state if str(key).startswith("loss_")} != {
            "loss_adversary", "loss_agent",
        }:
            raise ValueError(f"Native checkpoint loss-group set is not exactly PCP: {path}.")
        states.append(state)
        evidence[label] = {
            "path": str(path.resolve()), "sha256": sha256(path),
            "total_frames": header["total_frames"],
            "n_iters_performed": header["n_iters_performed"],
        }
    count, unequal, metadata_count, unequal_metadata = 0, [], 0, []
    for group in ("adversary", "agent"):
        key = f"loss_{group}"
        left, right = states[0][key], states[1][key]
        required_metadata = {f"{network}_network_params.{field}"
                             for network in ("actor", "critic")
                             for field in ("__batch_size", "__device")}
        if any(not isinstance(value, Mapping) or not required_metadata.issubset(value)
               for value in (left, right)):
            raise ValueError(f"480k loss-state or TensorDict metadata is missing for {group}.")
        if set(left) != set(right):
            raise ValueError(f"480k loss-state key sets differ for {group}.")
        for name in left:
            if not isinstance(left[name], torch.Tensor) or not isinstance(
                right[name], torch.Tensor
            ):
                if not name.endswith((".__batch_size", ".__device")):
                    raise ValueError(f"Unsupported non-tensor loss-state item: {group}/{name}.")
                metadata_count += 1
                if type(left[name]) is not type(right[name]) or left[name] != right[name]:
                    unequal_metadata.append(f"{group}/{name}")
                continue
            count += 1
            a, b = left[name], right[name]
            if (
                a.dtype != b.dtype
                or a.shape != b.shape
                or not torch.equal(
                    a.contiguous().reshape(-1).view(torch.uint8),
                    b.contiguous().reshape(-1).view(torch.uint8),
                )
            ):
                unequal.append(f"{group}/{name}")
    evidence.update(
        tensor_count=count,
        unequal_tensors=unequal,
        metadata_count=metadata_count,
        unequal_metadata=unequal_metadata,
        equal=not unequal and not unequal_metadata,
    )
    return evidence


def compare(original_suite, repaired_suite, through):
    if type(through) is not int or through != PREFIX_FRAMES:
        raise ValueError(f"This protocol-bound comparison requires exactly {PREFIX_FRAMES} frames.")
    if original_suite.resolve() == repaired_suite.resolve():
        raise ValueError("Original and repaired suites must be distinct.")
    originals, repaired = runs_by_seed(original_suite), runs_by_seed(repaired_suite)
    results = []
    for seed in (10, 11):
        old, old_metadata = originals[seed]
        new, new_metadata = repaired[seed]
        issues = []
        for label, run, metadata, expected in (
            ("original", old, old_metadata, "failed"),
            ("repaired", new, new_metadata, "completed"),
        ):
            if (
                json.loads((run / "status.json").read_text())["status"] != expected
                or metadata.get("status") != expected
            ):
                issues.append(f"{label} metadata/status do not both identify {expected}.")
            source = metadata.get("source_sha256")
            if not isinstance(source, str) or len(source) != 64 or any(
                letter not in "0123456789abcdef" for letter in source
            ):
                issues.append(f"{label} source fingerprint is missing or malformed.")
        if new_metadata.get("frames") != 600000:
            issues.append("Repaired completed run does not record the full 600k budget.")
        if scientific_spec(old) != scientific_spec(new):
            issues.append("Full resolved specifications differ beyond output placement.")
        if old_metadata["source_sha256"] == new_metadata["source_sha256"]:
            issues.append("Repaired cohort does not identify a distinct implementation source.")
        if (
            not isinstance(old_metadata.get("versions"), Mapping)
            or not old_metadata["versions"]
            or old_metadata.get("versions") != new_metadata.get("versions")
        ):
            issues.append("Saved package versions differ.")
        for field in ("sampling_device", "train_device", "buffer_device", "torch_num_threads"):
            if old_metadata["runtime"][field] != new_metadata["runtime"][field]:
                issues.append(f"Saved runtime {field} differs.")
        old_values, old_excluded = canonical_metrics(old / "metrics.csv", through)
        new_values, new_excluded = canonical_metrics(new / "metrics.csv", through)
        missing = sorted(set(old_values) - set(new_values))
        extra = sorted(set(new_values) - set(old_values))
        unequal = sorted(
            key
            for key in old_values.keys() & new_values.keys()
            if old_values[key] != new_values[key]
        )
        if missing or extra or unequal:
            issues.append("Recorded scientific prefix differs.")
        checkpoints = checkpoint_parameters(old, new)
        if not checkpoints["equal"]:
            issues.append("480k actor/critic loss-state tensors differ.")
        results.append(
            {
                "seed": seed,
                "original_run": str(old.resolve()),
                "repaired_run": str(new.resolve()),
                "original_source_sha256": old_metadata["source_sha256"],
                "repaired_source_sha256": new_metadata["source_sha256"],
                "original_metrics_sha256": sha256(old / "metrics.csv"),
                "repaired_metrics_sha256": sha256(new / "metrics.csv"),
                "compared_metric_count": len(old_values),
                "retained_original": retained_hashes(old_values),
                "retained_repaired": retained_hashes(new_values),
                "excluded_original": old_excluded,
                "excluded_repaired": new_excluded,
                "missing_count": len(missing),
                "extra_count": len(extra),
                "unequal_count": len(unequal),
                "first_differences": [
                    {
                        "key": key,
                        "original_hex": old_values.get(key),
                        "repaired_hex": new_values.get(key),
                    }
                    for key in (missing + extra + unequal)[:25]
                ],
                "checkpoint_480k": checkpoints,
                "checks_passed": not issues,
                "issues": issues,
            }
        )
    return {
        "schema_version": 1,
        "through_frames": through,
        "comparison": "Exact canonical numeric metric equality and 480k loss-state tensor equality",
        "excluded": ["timestamp column", "wall_time_seconds metric", *sorted(REPLAY_METRICS)],
        "interpretation": "Repair invariance only; original failed attempts remain failed.",
        "script_sha256": sha256(Path(__file__)),
        "checks_passed": all(row["checks_passed"] for row in results),
        "runs": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-suite", required=True, type=Path)
    parser.add_argument("--repaired-suite", required=True, type=Path)
    parser.add_argument(
        "--through-frames", type=int, choices=[PREFIX_FRAMES], default=PREFIX_FRAMES
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(args.out.resolve().is_relative_to(suite.resolve())
           for suite in (args.original_suite, args.repaired_suite)):
        parser.error("--out must be outside both managed suites to preserve the compared evidence.")
    try:
        report = compare(args.original_suite, args.repaired_suite, args.through_frames)
    except (ValueError, OSError, KeyError, TypeError) as error:
        report = {"schema_version": 1, "checks_passed": False, "issues": [str(error)]}
    atomic_write_json(args.out, report)
    print(json.dumps({"checks_passed": report["checks_passed"], "report": str(args.out)}))
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
