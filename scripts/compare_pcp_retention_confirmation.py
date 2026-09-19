"""Audit the separately declared CPUv3 retention repair without waiving old evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from commstudy.analysis.diagnostics import rebuild_spec
from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.config import resolved_experiment_dict, scientific_config_sha256
from commstudy.experiments.protocols import content_sha256, load_protocol, protocol_spec
from commstudy.experiments.provenance import source_fingerprint

SOURCE = "7124976af366d28d40a3ccf0ebce62aa2b4019783d13974b9f98dd1c316af262"
ORIGINAL_SOURCE = "73bc2cb332c0d4ccd5c6afb893b1e929f385b4cabfc6cb743c99eb6d0ec9d90c"
LEGACY_SHA = "44d2ac34c675ca8dfdc0f8dc33bd14a7676b536d6416bf44cf128c4ca91e1e9f"
CHECKPOINT_FRAMES = (120000, 240000, 360000, 480000, 600000)
GROUPS = ("adversary", "agent")
HISTORICAL_PROTOCOL_HASHES = {
    1: "d61d39f0e47a234e47a7fd6c3c3bea2c3ece91f586917e284608fa938e840bd0",
    2: "6f99321ab3b89222dccf2140183f8c387d3268d174d66bca78a6bbcc88a9c87a",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_module():
    path = Path(__file__).with_name("compare_pcp_confirmation_prefix.py")
    need(sha(path) == LEGACY_SHA, "Immutable original prefix comparator changed.")
    spec = importlib.util.spec_from_file_location("pcp_original_prefix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_spec(run, *, expected_keep):
    spec = rebuild_spec(run)
    document = resolved_experiment_dict(spec)
    need(
        type(document["experiment"].get("keep_checkpoints_num")) is type(expected_keep)
        and document["experiment"].get("keep_checkpoints_num") == expected_keep
        and "keep_checkpoints_num" in document["experiment"],
        f"{run}: checkpoint retention does not match the declared version.",
    )
    document["experiment"].pop("save_folder", None)
    document["experiment"].pop("keep_checkpoints_num")
    return spec, document


def complete_metrics(path):
    """All 600k scalar values, including replay diagnostics, except wall time."""
    selected, seen, excluded, collection_frames = {}, set(), {}, set()
    columns = {"timestamp", "frames", "iteration", "phase", "group", "metric", "sample", "value"}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        need(
            set(reader.fieldnames or []) == columns and len(reader.fieldnames) == len(columns),
            f"Unspecified or missing CSV columns: {path}.",
        )
        for row in reader:
            frame, iteration = int(row["frames"]), int(row["iteration"])
            value = float(row["value"])
            need(
                0 <= frame <= 600000 and iteration >= 0 and math.isfinite(value),
                f"Invalid frame/iteration/value in {path}.",
            )
            key = (
                str(frame),
                str(iteration),
                row["phase"],
                row["group"],
                row["metric"],
                row["sample"],
            )
            need(key not in seen, f"Duplicate metric key in {path}: {key}.")
            seen.add(key)
            if row["phase"] == "collection":
                collection_frames.add(frame)
            if row["metric"] == "wall_time_seconds":
                excluded["wall_time_seconds"] = excluded.get("wall_time_seconds", 0) + 1
            else:
                selected[key] = value.hex()
    need(
        collection_frames == set(range(6000, 600001, 6000)),
        f"Collection records do not cover exactly all 100 batches: {path}.",
    )
    need(bool(selected), f"No retained metric values in {path}.")
    return selected, excluded


def compare_values(left, right, left_excluded, right_excluded, helper):
    missing, extra = sorted(left.keys() - right.keys()), sorted(right.keys() - left.keys())
    unequal = sorted(key for key in left.keys() & right.keys() if left[key] != right[key])
    return {
        "equal": not missing and not extra and not unequal,
        "compared_metric_count": len(left),
        "left_retained": helper.retained_hashes(left),
        "right_retained": helper.retained_hashes(right),
        "left_excluded": left_excluded,
        "right_excluded": right_excluded,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "unequal_count": len(unequal),
        "first_differences": [
            {"key": key, "left_hex": left.get(key), "right_hex": right.get(key)}
            for key in (missing + extra + unequal)[:25]
        ],
    }


def native_checkpoint(run, frames):
    paths = list(run.glob(f"benchmarl/*/checkpoints/checkpoint_{frames}.pt"))
    need(len(paths) == 1, f"Require exactly one retained {frames}-frame checkpoint in {run}.")
    path = paths[0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    need(
        isinstance(payload, Mapping) and isinstance(payload.get("state"), Mapping),
        f"Native checkpoint state/header missing: {path}.",
    )
    header = payload["state"]
    need(
        type(header.get("total_frames")) is int
        and header["total_frames"] == frames
        and type(header.get("n_iters_performed")) is int
        and header["n_iters_performed"] == frames // 6000,
        f"Native checkpoint frame/update header differs: {path}.",
    )
    need(
        {key for key in payload if str(key).startswith("loss_")} == {f"loss_{g}" for g in GROUPS},
        f"Native checkpoint loss groups differ: {path}.",
    )
    return path, payload


def compare_checkpoint(left_run, right_run, frames):
    left_path, left = native_checkpoint(left_run, frames)
    right_path, right = native_checkpoint(right_run, frames)
    unequal, tensor_count, metadata_count = [], 0, 0
    for group in GROUPS:
        a, b = left[f"loss_{group}"], right[f"loss_{group}"]
        required = {
            f"{network}_network_params.{field}"
            for network in ("actor", "critic")
            for field in ("__batch_size", "__device")
        }
        need(
            isinstance(a, Mapping)
            and isinstance(b, Mapping)
            and set(a) == set(b)
            and required.issubset(a),
            f"{group}: checkpoint loss-state keys or metadata differ.",
        )
        group_tensors = 0
        for name, value in a.items():
            other = b[name]
            if isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor):
                tensor_count += 1
                group_tensors += 1
                equal = (
                    value.dtype == other.dtype
                    and value.shape == other.shape
                    and torch.equal(
                        value.contiguous().reshape(-1).view(torch.uint8),
                        other.contiguous().reshape(-1).view(torch.uint8),
                    )
                )
            else:
                need(
                    name.endswith((".__batch_size", ".__device"))
                    and not isinstance(value, torch.Tensor)
                    and not isinstance(other, torch.Tensor),
                    f"Unsupported checkpoint metadata: {group}/{name}.",
                )
                need(
                    (
                        name.endswith(".__batch_size")
                        and type(value) is torch.Size
                        and type(other) is torch.Size
                    )
                    or (name.endswith(".__device") and value is None and other is None),
                    f"Invalid checkpoint metadata type/value: {group}/{name}.",
                )
                metadata_count += 1
                equal = type(value) is type(other) and value == other
            if not equal:
                unequal.append(f"{group}/{name}")
        need(group_tensors > 0, f"No actor/critic tensors for {group}.")
    return {
        "left": {"path": str(left_path.resolve()), "sha256": sha(left_path)},
        "right": {"path": str(right_path.resolve()), "sha256": sha(right_path)},
        "frames": frames,
        "updates": frames // 6000,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "unequal_keys": unequal,
        "equal": not unequal,
    }


def retention_manifest(run):
    files = list(run.glob("benchmarl/*/checkpoints/checkpoint_*.pt"))
    need(
        {p.name for p in files} == {f"checkpoint_{frame}.pt" for frame in CHECKPOINT_FRAMES}
        and len(files) == len(CHECKPOINT_FRAMES),
        "CPUv3 must preserve all five unique periodic checkpoints after its final save.",
    )
    paths = [native_checkpoint(run, frame)[0] for frame in CHECKPOINT_FRAMES]
    final = run / "checkpoints/policy_state.pt"
    need(final.is_file(), "CPUv3 final managed policy checkpoint is missing.")
    snapshots = list(run.glob("benchmarl/*/replay_diagnostics/*adversary_first_fallback*.pt"))
    need(len(snapshots) == 1, "CPUv3 first predator fallback snapshot is missing or ambiguous.")
    paths.extend([final, *snapshots])
    return {str(path.resolve()): sha(path) for path in paths}


def compare(original_suite, repaired_suite, retained_suite, protocol_path):
    helper = legacy_module()
    root = Path(__file__).resolve().parents[1]
    need(
        source_fingerprint(root) == SOURCE,
        "Run this comparison from the frozen CPUv3 execution copy.",
    )
    need(
        len({p.resolve() for p in (original_suite, repaired_suite, retained_suite)}) == 3,
        "All three cohort directories must be distinct.",
    )
    protocol = load_protocol(protocol_path)
    need(
        protocol["protocol_id"] == "pcp_corrected_cpu_v3"
        and protocol["base_spec"]["experiment"]["keep_checkpoints_num"] is None,
        "Expected the separately declared CPUv3 retention protocol.",
    )
    cohorts = [
        helper.runs_by_seed(path) for path in (original_suite, repaired_suite, retained_suite)
    ]
    results = []
    for seed in (10, 11):
        rows = [cohort[seed] for cohort in cohorts]
        paths = [
            run / name
            for run, _ in rows
            for name in ("metadata.json", "status.json", "resolved_config.yaml", "metrics.csv")
        ]
        paths.append(protocol_path)
        for (run, metadata), version, expected_status, source, keep in zip(
            rows,
            (1, 2, 3),
            ("failed", "completed", "completed"),
            (ORIGINAL_SOURCE, SOURCE, SOURCE),
            (2, 2, None),
            strict=True,
        ):
            need(
                metadata.get("status") == expected_status
                and json.loads((run / "status.json").read_text()).get("status") == expected_status,
                f"CPUv{version} managed status differs from {expected_status}.",
            )
            need(
                metadata.get("source_sha256") == source
                and metadata.get("protocol", {}).get("source_sha256") == source
                and metadata["protocol"].get("protocol_id") == f"pcp_corrected_cpu_v{version}"
                and metadata["protocol"].get("stage") == "confirmation"
                and metadata.get("model") == "pcp_comm_identity",
                f"CPUv{version} source/protocol binding differs.",
            )
            if version in HISTORICAL_PROTOCOL_HASHES:
                need(
                    metadata["protocol"].get("protocol_sha256")
                    == HISTORICAL_PROTOCOL_HASHES[version],
                    f"CPUv{version} historical protocol hash differs.",
                )
            spec, document = normalized_spec(run, expected_keep=keep)
            need(
                metadata.get("scientific_config_sha256") == scientific_config_sha256(spec),
                f"CPUv{version} scientific configuration hash differs.",
            )
            if version == 1:
                reference = document
            else:
                need(metadata.get("frames") == 600000, f"CPUv{version} did not finish 600k.")
                need(
                    document == reference,
                    "Resolved settings differ beyond retention and output placement.",
                )
            if version == 3:
                expected_spec = protocol_spec(protocol, model="pcp_comm_identity", seed=seed)
                expected_document = resolved_experiment_dict(expected_spec)
                expected_document["experiment"].pop("save_folder", None)
                expected_document["experiment"].pop("keep_checkpoints_num")
                need(
                    document == expected_document
                    and metadata["protocol"].get("protocol_sha256") == content_sha256(protocol),
                    "CPUv3 saved specification/protocol hash differs from its declaration.",
                )
        runtime = protocol["runtime"]
        for _, metadata in rows:
            need(
                metadata.get("versions") == {"python": runtime["python"], **runtime["packages"]},
                "Saved software versions differ from the frozen runtime.",
            )
            need(
                all(
                    metadata.get("runtime", {}).get(k) == "cpu"
                    for k in ("sampling_device", "train_device", "buffer_device")
                )
                and metadata["runtime"].get("torch_num_threads") == 1,
                "Saved device/thread runtime differs.",
            )
        old, previous, retained = (row[0] for row in rows)
        manifest = retention_manifest(retained)
        paths.extend(Path(path) for path in manifest)
        paths.extend([native_checkpoint(old, 480000)[0], native_checkpoint(previous, 600000)[0]])
        before = {str(path.resolve()): sha(path) for path in paths}
        old_values, old_excluded = helper.canonical_metrics(old / "metrics.csv", 558000)
        new_values, new_excluded = helper.canonical_metrics(retained / "metrics.csv", 558000)
        prefix = compare_values(old_values, new_values, old_excluded, new_excluded, helper)
        previous_values, previous_excluded = complete_metrics(previous / "metrics.csv")
        retained_values, retained_excluded = complete_metrics(retained / "metrics.csv")
        full = compare_values(
            previous_values, retained_values, previous_excluded, retained_excluded, helper
        )
        checkpoint_480k = compare_checkpoint(old, retained, 480000)
        checkpoint_600k = compare_checkpoint(previous, retained, 600000)
        after = {str(path.resolve()): sha(path) for path in paths}
        need(before == after, "Audited inputs changed during retention comparison.")
        results.append(
            {
                "seed": seed,
                "checks_passed": all(
                    item["equal"] for item in (prefix, full, checkpoint_480k, checkpoint_600k)
                ),
                "original_v1_vs_v3_prefix_through558k": prefix,
                "repaired_v2_vs_v3_full600k": full,
                "v1_vs_v3_checkpoint_480k": checkpoint_480k,
                "v2_vs_v3_checkpoint_600k": checkpoint_600k,
                "retained_artifact_manifest": manifest,
                "artifact_hashes_before": before,
                "artifact_hashes_after": after,
            }
        )
    return {
        "schema_version": 1,
        "comparison": "pcp_retention_confirmation_v3",
        "checks_passed": all(row["checks_passed"] for row in results),
        "runs": results,
        "allowed_resolved_differences": {
            "save_folder": "output placement",
            "keep_checkpoints_num": {"v1": 2, "v2": 2, "v3": None},
        },
        "v1_prefix_exclusions": [
            "timestamp column",
            "wall_time_seconds metric",
            *sorted(helper.REPLAY_METRICS),
        ],
        "v2_full_exclusions": ["timestamp column", "wall_time_seconds metric"],
        "source_sha256": SOURCE,
        "protocol_sha256": content_sha256(protocol),
        "script_sha256": sha(Path(__file__)),
        "legacy_script_sha256": LEGACY_SHA,
        "approval_decision": "not_issued",
        "limits": "Retention and observed-trajectory invariance only. Original failed v1 and "
        "unconfirmed v2 outcomes remain unchanged. Full final-policy and scientific review "
        "are separate requirements.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-suite", "repaired-suite", "retained-suite", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    suites = (args.original_suite, args.repaired_suite, args.retained_suite)
    need(
        not any(args.out.resolve().is_relative_to(path.resolve()) for path in suites),
        "Derived output must remain outside all audited cohorts.",
    )
    try:
        report = compare(*suites, args.protocol)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        report = {
            "schema_version": 1,
            "checks_passed": False,
            "issues": [str(error)],
            "approval_decision": "not_issued",
        }
    atomic_write_json(args.out, report)
    print(json.dumps({"checks_passed": report["checks_passed"], "report": str(args.out)}))
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
