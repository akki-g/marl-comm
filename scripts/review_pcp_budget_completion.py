"""Independently review a closed D4 cohort; never train, reconstruct policies or select actors.

Importing this module uses the standard library only. NumPy and Torch are loaded
only after the real audit has rejected incomplete queue/report evidence. Statistics
are recomputed from raw episode records without importing the frozen analyzer.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import stat
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("/Users/akshatguduru/Desktop/Thesis/marl-comm")
SUITE = "pcp_visibility_budget_cpu_v1"
SOURCE = "69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e"
PROTOCOL = "ec7c109413625f270b79bffc1753adf5b7691d008eb90a4297d844b020898b0a"
PLAN = "5b1474bca7f4277efc71fa3ec7f2359c915a6cf5ebf71e1e1d356db51b7db4ba"
MODELS = ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0", "pcp_broadcast_k3")
VISIBILITIES = ("radius1", "global")
SEEDS = tuple(range(20, 25))
EPISODES = list(range(200000, 200256))
ARMS = ("live", "severed", "suppress_0", "suppress_1", "suppress_2", "shuffle")
PRIMARY = "broadcast_k3_minus_pcp_local_capacity/radius1"
FROZEN_FILES = {
    "docs/PCP_VISIBILITY_BUDGET_PLAN.md": PLAN,
    "configs/protocols/pcp_visibility_budget_cpu_v1_frozen.yaml": (
        "d8b6feea2daa83912e83f2525db1038115391a34bd87e18ef72527b1e30372b0"
    ),
    "scripts/analyze_pcp_budget.py": (
        "f0cc0ed507c6ca51ccbb8db83ed0bc0bd642c85836a1dbdf03032c89bb02854f"
    ),
    "scripts/evaluate_pcp_budget.py": (
        "223d03a1ac5368d7d9ed6db032ac3eaa424e979dae27e42b2b19fbda36768594"
    ),
    "scripts/archive_pcp_budget_run.py": (
        "36c9458f81a0477cbdc3cc31971ab3c92af19ea16d5a955c8fdca5fad3245e1f"
    ),
}
HISTORICAL = {
    "pcp_comm_identity": (
        "9e0120d070db3a37723fc440a8e9ef5dbd90c7bb23288147ff603ba3c10593b1",
        "43681456fe9096e2faa457369b389eb25d549368dbe6e925bbca38c716647cc7",
    ),
    "pcp_local_capacity": (
        "fbed25c0522a9b67fcd4f701b20603bbcac561206d5eed7a82701ed5167e9a8e",
        "6a0e1934f1a90782e90ef3e41d210d5d8d115e29796cb01b4dbf3949320ade5a",
    ),
}
OUTPUTS = {
    "analysis_report.json",
    "per_seed.csv",
    "episode_metrics.csv",
    "training_curves.csv",
    *(
        f"{name}.{extension}"
        for name in ("training_returns", "heldout_outcomes", "tested_budget_frontier")
        for extension in ("pdf", "svg", "png")
    ),
}
DOMAIN = {
    "episode_steps",
    "complete",
    "terminated",
    "truncated",
    "return",
    "any_contact",
    "first_contact_observed",
    "first_contact_step_or_horizon",
    "contact_duration_steps",
    "contact_pair_steps",
    "simultaneous_contact_steps",
    "any_simultaneous_contact",
    "contacting_predators_max",
    "contacting_predators_mean",
    "mean_predator_nearest_prey_distance",
    "nearest_predator_prey_distance_mean",
    "nearest_predator_prey_distance_min",
    "predator_boundary_fraction",
    "prey_boundary_fraction",
    "predator_obstacle_collision_pair_steps",
    "predator_teammate_collision_pair_steps",
    "sender_visible_receiver_blind_pairs_mean",
    "sender_visible_receiver_blind_prey_triples_mean",
    "reward_accounting_max_abs_error",
    "contact_by_deadline",
    *(f"prey_visible_to_{k}_predators_mean" for k in range(4)),
}
HEALTH = {
    "action_scalars",
    "all_finite",
    "saturation_fraction",
    "max_abs_action",
    "max_abs_loc",
    "min_scale",
    "max_scale",
}
RUN_INPUTS = (
    "metadata.json",
    "status.json",
    "resolved_config.yaml",
    "metrics.csv",
    "checkpoints/policy_state.pt",
)
PAIR_KEYS = ("episode_id", "seed", "initial_physical_state_sha256", "exogenous_initial")
CONDITION_CIS = (
    "contact_by_deadline",
    "return",
    "contact_duration_steps",
    "simultaneous_contact_steps",
    "predator_boundary_fraction",
    "prey_boundary_fraction",
    "action_saturation_fraction",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def typed_fields(actual, expected):
    return all(type(actual.get(key)) is type(value) and actual[key] == value
               for key, value in expected.items())


def ordinary(path, *, directory=False):
    path = Path(path)
    need(
        path.is_absolute() and ".." not in path.parts, f"Require an absolute ordinary path: {path}"
    )
    need(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlink input: {path}")
    need(path.is_dir() if directory else path.is_file(), f"Missing ordinary input: {path}")
    return path


def digest(path):
    with ordinary(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def unique_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def floating(value):
        result = float(value)
        need(math.isfinite(result), "Nonfinite JSON number.")
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(
        payload, object_pairs_hook=pairs, parse_float=floating, parse_constant=constant
    )


class Inputs:
    """Retain the first observed binding; never replace it with changed bytes."""

    def __init__(self):
        self.files = {}
        self.trees = {}

    def pin(self, path, expected=None):
        path = ordinary(path)
        record = {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        need(expected is None or record["sha256"] == expected, f"Input hash differs: {path}")
        need(
            str(path) not in self.files or self.files[str(path)] == record,
            f"Input changed after its first binding: {path}",
        )
        self.files[str(path)] = record
        return record["sha256"]

    def read(self, path, expected=None):
        self.pin(path, expected)
        return unique_json(Path(path).read_text())

    def tree(self, root):
        files = regular_tree(root)
        names = sorted(files)
        need(
            str(root) not in self.trees or self.trees[str(root)] == names,
            f"Raw file inventory changed: {root}",
        )
        self.trees[str(root)] = names
        return files

    def finish(self):
        for root, names in list(self.trees.items()):
            need(sorted(regular_tree(Path(root))) == names, f"Raw file inventory changed: {root}")
        for path, record in list(self.files.items()):
            self.pin(Path(path), record["sha256"])


def regular_tree(root):
    root = ordinary(root, directory=True)
    files = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        need(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f"Nonordinary raw evidence: {path}")
        if stat.S_ISREG(mode):
            files[path.relative_to(root).as_posix()] = path
    need(files, f"Empty raw evidence: {root}")
    return files


def package_snapshot(root, inputs):
    package = root / "src/commstudy"
    paths = sorted(ordinary(package, directory=True).rglob("*.py"))
    hasher = hashlib.sha256()
    for path in paths:
        inputs.pin(path)
        hasher.update(path.relative_to(package).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    need(len(paths) == 54 and hasher.hexdigest() == SOURCE, "Frozen 54-file package changed.")
    freeze = inputs.read(root / "results" / SUITE / "package_source_freeze.json")
    actual = {p.relative_to(root).as_posix(): inputs.files[str(p)]["sha256"] for p in paths}
    need(
        freeze["source_sha256"] == SOURCE and freeze["files"] == actual,
        "Package inventory differs from its freeze receipt.",
    )
    return set(actual)


def original(row):
    return row["model"] in HISTORICAL and row["visibility"] == "radius1" and row["seed"] == 20


def expected_run_id(model, visibility, seed):
    return (
        f"predator_capture_prey__mappo__{model}__{SUITE}__visibility__{visibility}__seed{seed:03d}"
    )


def closed_cohort(report, queue):
    """This gate is deliberately before any NumPy/Torch import or row inspection."""
    need(
        type(report.get("schema_version")) is int and report["schema_version"] == 1
        and report.get("analysis") == "pcp_visibility_budget_seed_analysis_v1"
        and report.get("checks_passed") is True
        and report.get("issues") == []
        and report.get("status") == "complete_for_review"
        and report.get("expected_policies") == report.get("valid_policies") == 40
        and report.get("approval_decision") == "not_issued",
        "Require the complete frozen forty-policy analyzer report.",
    )
    need(
        report.get("training_seeds") == list(SEEDS)
        and report.get("heldout_episode_seeds") == EPISODES
        and report.get("deadline") == 50
        and report.get("target") == 0.8
        and report.get("protocol_sha256") == PROTOCOL
        and report.get("plan_sha256") == PLAN
        and report.get("training_source_sha256") == report.get("analysis_source_sha256") == SOURCE
        and report.get("primary_contrast") == PRIMARY,
        "Frozen report identity differs.",
    )
    rows = report["rows"]
    need(
        len(rows) == 40
        and {(r["model"], r["visibility"], r["seed"]) for r in rows}
        == set(itertools.product(MODELS, VISIBILITIES, SEEDS)),
        "Incomplete or duplicate grid.",
    )
    need(
        queue.get("status") == "completed"
        and queue.get("phase") == "final_analysis"
        and set(queue["rows"]) == {r["run_id"] for r in rows},
        "Queue has not fully closed.",
    )
    for row in rows:
        need(
            row["run_id"] == expected_run_id(row["model"], row["visibility"], row["seed"])
            and row["status"] == "valid"
            and row["validation_passed"] is True
            and row["issues"] == [],
            "Invalid completed policy identity.",
        )
        q = queue["rows"][row["run_id"]]
        need(
            q["status"] == "completed"
            and q["stage"] == "complete"
            and q["training_completed"] is True
            and q["evaluation_checks_passed"] is True
            and q["archive"]["checks_passed"] is True
            and q["training_reused"] is original(row),
            "Queue row has not closed every preservation gate.",
        )


def manifest_rows(path, report, inputs):
    inputs.pin(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    need(
        len(rows) == 40
        and len({r["run_id"] for r in rows}) == 40
        and {r["run_id"] for r in rows} == {r["run_id"] for r in report["rows"]},
        "Analyzed manifest is not the complete unique cohort.",
    )
    result = {}
    for row in rows:
        need(
            row["run_id"] == expected_run_id(row["model"], row["ablation_value"], int(row["seed"]))
            and row["suite_id"] == SUITE
            and row["task"] == "vmas_predator_capture_prey"
            and row["algorithm"] == "mappo"
            and row["ablation"] == "visibility"
            and row["max_n_frames"] == "600000"
            and row["attempt"] == "0"
            and row["retry_of"] == "",
            "Manifest scientific identity differs.",
        )
        protocol = unique_json(row["protocol_json"])
        need(
            protocol["source_sha256"] == SOURCE
            and protocol["sha256"] == PROTOCOL
            and protocol["stage"] == "comparison",
            "Manifest protocol binding differs.",
        )
        row["protocol"] = protocol
        result[row["run_id"]] = row
    return result


def archive_inventory(path):
    """Stream regular members without extraction, then consume the gzip checksum trailer."""
    observed, inventory, embedded_sha = {}, None, None
    with tarfile.open(ordinary(path), "r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            need(
                member.isfile()
                and not name.is_absolute()
                and ".." not in name.parts
                and name.as_posix() == member.name
                and member.name not in observed,
                f"Unsafe or duplicate archive member: {member.name}",
            )
            stream = archive.extractfile(member)
            need(stream is not None, "Unreadable archive member.")
            if member.name == "ARCHIVE_MANIFEST.json":
                need(member.size <= 16 * 1024 * 1024, "Unexpectedly large archive manifest.")
                payload = stream.read()
                inventory = unique_json(payload)
                value = embedded_sha = hashlib.sha256(payload).hexdigest()
            else:
                value = hashlib.file_digest(stream, "sha256").hexdigest()
            observed[member.name] = {"sha256": value, "bytes": member.size, "mode": member.mode}
    need(
        isinstance(inventory, dict)
        and type(inventory.get("schema_version")) is int and inventory["schema_version"] == 1,
        "Missing archive inventory.",
    )
    need(
        set(observed) == set(inventory["files"]) | {"ARCHIVE_MANIFEST.json"},
        "Archive member inventory differs.",
    )
    for name, entry in inventory["files"].items():
        need(
            type(entry["bytes"]) is int and type(entry["mode"]) is int
            and observed[name] == {k: entry[k] for k in ("sha256", "bytes", "mode")},
            f"Archive readback differs: {name}",
        )
    with gzip.open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    return inventory, embedded_sha


def mapped_source(path, root, historical):
    path = Path(path)
    base = OLD_ROOT if historical else root
    need(
        path.is_absolute() and ".." not in path.parts and path.is_relative_to(base),
        f"Unexpected archive source prefix: {path}",
    )
    return root / path.relative_to(base)


def normalize_native(native, root, historical):
    result = {
        **native,
        "checkpoints": [dict(r) for r in native["checkpoints"]],
        "replay_diagnostics": [dict(r) for r in native["replay_diagnostics"]],
    }
    for row in [*result["checkpoints"], *result["replay_diagnostics"]]:
        row["path"] = str(mapped_source(row["path"], root, historical))
    return result


def verify_run_archive(root, row, q, plan, manifest, queue_dir, package_files, inputs):
    run = root / "runs" / SUITE / row["run_id"]
    need(
        Path(row["run_directory"]) == run
        and Path(plan["output_root"]) / SUITE / row["run_id"] == run,
        "Run is outside the frozen managed suite.",
    )
    raw = inputs.tree(run)
    archived = q["archive"]
    destination = Path(archived["destination"])
    need(destination == queue_dir / "run_archives" / row["run_id"], "Archive destination differs.")
    receipt_path, tar_path = destination / "archive_manifest.json", destination / "run.tar.gz"
    receipt = inputs.read(receipt_path, archived["receipt_sha256"])
    inputs.pin(tar_path, archived["archive_sha256"])
    historical = original(row)
    wanted_files = {"archive_manifest.json", "run.tar.gz"}
    if historical:
        wanted_files.add("archive_reuse_receipt.json")
    need(set(inputs.tree(destination)) == wanted_files, "Unexpected or failed archive attempt.")
    inventory, embedded_sha = archive_inventory(tar_path)
    expected = {
        "schema_version": 1,
        "archive": "pcp_budget_completed_run_v1",
        "run_id": row["run_id"],
        "suite_id": SUITE,
        "source_sha256": SOURCE,
        "protocol_sha256": PROTOCOL,
        "model": row["model"],
        "seed": row["seed"],
        "visibility": row["visibility"],
        "frames": 600000,
        "iterations": 100,
    }
    need(
        all(type(inventory.get(k)) is type(v) and inventory[k] == v for k, v in expected.items()),
        "Embedded archive policy identity differs.",
    )
    expected_receipt = {
        "schema_version": 1,
        "checks_passed": True,
        "run_id": row["run_id"],
        "archive_file": "run.tar.gz",
        "archive_sha256": archived["archive_sha256"],
        "archive_bytes": tar_path.stat().st_size,
        "source_before_after_equal": True,
        "all_archive_files_rehashed": True,
        "original_files_removed": False,
        "embedded_manifest_sha256": embedded_sha,
        "archived_files": len(inventory["files"]),
        "run_file_count": len(raw),
        "uncompressed_file_bytes": sum(r["bytes"] for r in inventory["files"].values()),
    }
    need(
        set(receipt) == set(expected_receipt) and typed_fields(receipt, expected_receipt)
        and type(archived["archive_bytes"]) is int
        and archived["archive_bytes"] == receipt["archive_bytes"],
        "Archive receipt differs from independent readback.",
    )
    special = {
        "worker/stdout.log",
        "provenance/worker_manifest.csv",
        "provenance/protocol.yaml",
        "provenance/plan.md",
        "provenance/archive_helper.py",
        "provenance/inventory_helper.py",
        "provenance/source_freeze.json",
        "provenance/launch_approval.json",
        "source/pyproject.toml",
        "source/uv.lock",
    }
    need(
        set(inventory["files"])
        == special | {f"source/{p}" for p in package_files} | {f"run/{name}" for name in raw},
        "Archive omits or adds raw/source evidence.",
    )
    need(inventory["run_file_count"] == len(raw), "Raw run inventory count differs.")
    mapped = {}
    for name, record in inventory["files"].items():
        path = mapped_source(record["source_path"], root, historical)
        inputs.pin(path, record["sha256"])
        need(path.stat().st_size == record["bytes"], "Archive source size differs.")
        if name.startswith("run/"):
            need(path == raw[name.removeprefix("run/")], "Archived raw path differs.")
        if name.startswith("source/"):
            need(path == root / name.removeprefix("source/"), "Archived research source differs.")
        mapped[name] = path
    fixed_sources = {
        "provenance/protocol.yaml": root / plan["protocol"]["path"],
        "provenance/plan.md": root / "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
        "provenance/archive_helper.py": root / "scripts/archive_pcp_budget_run.py",
        "provenance/inventory_helper.py": root / "scripts/evaluate_pcp_budget.py",
        "provenance/source_freeze.json": root / "results" / SUITE / "package_source_freeze.json",
    }
    need(
        all(mapped[name] == path for name, path in fixed_sources.items()),
        "Archived protocol/plan/helper source identity differs.",
    )
    metadata = inputs.read(run / "metadata.json")
    identity = {
        "status": "completed",
        "run_id": row["run_id"],
        "suite_id": SUITE,
        "model": row["model"],
        "seed": row["seed"],
        "ablation": "visibility",
        "ablation_value": row["visibility"],
        "frames": 600000,
        "iterations": 100,
        "source_sha256": SOURCE,
        "scientific_config_sha256": plan["scientific_config_sha256"],
    }
    need(all(metadata.get(k) == v for k, v in identity.items()), "Raw training identity differs.")
    approval = inputs.read(mapped["provenance/launch_approval.json"])
    need(
        metadata["protocol"]["approval_sha256"] == canonical(approval),
        "Saved metadata no longer binds its original launch approval.",
    )
    need(metadata["parameters"] == row["parameters"], "Reported parameter counts differ.")
    need(
        f"[sweep] {row['run_id']} -> completed" in mapped["worker/stdout.log"].read_text(),
        "Archived worker log lacks successful closure.",
    )
    if historical:
        archive_sha, receipt_sha = HISTORICAL[row["model"]]
        need(
            archived["archive_sha256"] == archive_sha
            and archived["receipt_sha256"] == receipt_sha
            and archived.get("reused_original_archive") is True,
            "Historical archive differs from the immutable recovery anchors.",
        )
        reuse = inputs.read(
            destination / "archive_reuse_receipt.json", archived["reuse_receipt_sha256"]
        )
        expected_reuse = {
            "schema_version": 1,
            "checks_passed": True,
            "run_id": row["run_id"],
            "destination": str(destination),
            "original_files_removed": False,
            "historical_files_modified": False,
            "all_copied_files_rehashed": True,
            "copied_archive_readback_verified": True,
        }
        need(
            typed_fields(reuse, expected_reuse),
            "Archive reuse receipt differs.",
        )
        source = reuse["source_verification"]
        origin = root / "results" / SUITE / "interim_hold_archives" / row["run_id"]
        need(set(inputs.tree(origin)) == {"run.tar.gz", "archive_manifest.json"},
             "Original archive inventory differs.")
        need(
            source["source"] == str(origin)
            and source["run_id"] == row["run_id"]
            and source["checks_passed"] is True
            and source["original_approval_matches_metadata"] is True
            and source["all_archive_files_rehashed"] is True
            and source["source_before_after_equal"] is True
            and source["archive_sha256"] == archive_sha
            and source["receipt_sha256"] == receipt_sha
            and source["embedded_manifest_sha256"] == embedded_sha
            and source["archived_files"] == len(inventory["files"])
            and source["run_file_count"] == len(raw)
            and source["archive_bytes"] == tar_path.stat().st_size,
            "Original reuse source differs.",
        )
        inputs.pin(origin / "run.tar.gz", archive_sha)
        inputs.pin(origin / "archive_manifest.json", receipt_sha)
        binding = reuse["operational_binding_sha256"]
        need(
            binding.get(str(manifest)) == inputs.files[str(manifest)]["sha256"]
            and len(binding) == 2,
            "Reuse omits its separate operational manifest binding.",
        )
        relocation_path = next(Path(p) for p in binding if p != str(manifest))
        relocation = inputs.read(relocation_path, binding[str(relocation_path)])
        need(
            relocation_path == manifest.parent / "relocation_receipt.json"
            and relocation["checks_passed"] is True
            and relocation["source_sha256"] == SOURCE
            and relocation["generated_file_sha256"][manifest.name] == binding[str(manifest)],
            "Reuse relocation receipt differs.",
        )
        source_hashes = source["input_artifact_sha256"]
        old_log = root / "results" / SUITE / "training_queue" / f"{row['run_id']}.log"
        expected_hashes = {
            str(path): inventory["files"][name]["sha256"] for name, path in mapped.items()
        }
        expected_hashes.update(
            {
                str(origin / "run.tar.gz"): archive_sha,
                str(origin / "archive_manifest.json"): receipt_sha,
                str(old_log): inventory["files"]["worker/stdout.log"]["sha256"],
            }
        )
        need(source_hashes == expected_hashes, "Reuse source/worker-log hash inventory differs.")
        for path, value in source_hashes.items():
            inputs.pin(Path(path), value)
    else:
        need(
            archived.get("reused_original_archive", False) is False
            and mapped["provenance/worker_manifest.csv"] == manifest
            and mapped["worker/stdout.log"] == queue_dir / f"{row['run_id']}.log"
            and mapped["provenance/launch_approval.json"] == root / plan["protocol"]["approval"],
            "New archive has an unexpected manifest, log, approval or reuse binding.",
        )
    native = normalize_native(inventory["native_inventory"], root, historical)
    need(
        native["managed_final_policy_sha256"]
        == inventory["files"]["run/checkpoints/policy_state.pt"]["sha256"],
        "Archived managed actor identity differs.",
    )
    return (
        metadata,
        native,
        {
            "run_id": row["run_id"],
            "historical_archive_reused": historical,
            "archive_sha256": receipt["archive_sha256"],
            "archive_members": len(inventory["files"]) + 1,
            "raw_run_files": len(raw),
            "original_approval_preserved": True,
        },
    )


def finite(value):
    return type(value) in (int, float, bool) and math.isfinite(value)


def close(actual, expected, label):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ: {label}")
        for key, value in expected.items():
            close(actual[key], value, f"{label}/{key}")
    elif type(expected) in (int, float):
        need(
            finite(actual)
            and type(actual) is not bool
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"Recomputed value differs: {label}",
        )
    else:
        need(type(actual) is type(expected) and actual == expected, f"Value differs: {label}")


def verify_episode(episode, seed, model, arm):
    metrics, health, costs = episode["metrics"], episode["action_health"], episode["costs"]
    need(
        episode["episode_id"] == f"pcp-heldout:{seed}" and episode["seed"] == seed,
        "Episode identity/order differs.",
    )
    state = episode["initial_physical_state_sha256"]
    need(
        isinstance(state, str) and len(state) == 64 and all(c in "0123456789abcdef" for c in state),
        "Invalid physical state hash.",
    )
    exogenous = episode["exogenous_initial"]
    need(
        set(exogenous) == {"exogenous_episode_ids", "exogenous_steps"}
        and exogenous["exogenous_steps"] == [0]
        and len(exogenous["exogenous_episode_ids"]) == 1
        and type(exogenous["exogenous_episode_ids"][0]) is int
        and exogenous["exogenous_episode_ids"][0] >= 0,
        "Invalid exogenous identity.",
    )
    need(
        set(metrics) == DOMAIN
        and set(health) == HEALTH
        and all(finite(v) for v in [*metrics.values(), *health.values()]),
        "Missing or nonfinite episode measurements.",
    )
    need(
        metrics["episode_steps"] == 100
        and metrics["complete"] == metrics["truncated"] == 1
        and metrics["terminated"] == 0,
        "Episode is not a complete 100-transition truncation.",
    )
    d, p, s = (
        metrics[k]
        for k in ("contact_duration_steps", "contact_pair_steps", "simultaneous_contact_steps")
    )
    first = metrics["first_contact_step_or_horizon"]
    need(
        all(
            float(metrics[k]).is_integer()
            for k in (
                "contact_duration_steps",
                "contact_pair_steps",
                "simultaneous_contact_steps",
                "first_contact_step_or_horizon",
                "contacting_predators_max",
                "predator_obstacle_collision_pair_steps",
                "predator_teammate_collision_pair_steps",
            )
        ),
        "Fractional transition count.",
    )
    need(
        0 <= s <= d <= 100
        and d + s <= p <= d + 2 * s
        and 1 <= first <= 100
        and (d > 0 or first == 100)
        and (d == 0 or d <= 101 - first),
        "Impossible contact/censoring exposure.",
    )
    need(
        metrics["any_contact"] == metrics["first_contact_observed"] == int(d > 0)
        and metrics["any_simultaneous_contact"] == int(s > 0)
        and metrics["contact_by_deadline"] == int(d > 0 and first <= 50)
        and metrics["reward_accounting_max_abs_error"] == 0
        and metrics["return"] == 10 * p,
        "Deadline or repeated-contact accounting differs.",
    )
    maximum = 3 if p - d - s > 0 else 2 if s else 1 if d else 0
    need(
        metrics["contacting_predators_max"] == maximum
        and abs(metrics["contacting_predators_mean"] - p / 100) <= 1e-6,
        "Contact predator count differs.",
    )
    bins = [metrics[f"prey_visible_to_{k}_predators_mean"] for k in range(4)]
    need(
        all(0 <= v <= 1 for v in bins)
        and abs(sum(bins) - 1) <= 1e-6
        and all(
            0 <= metrics[k] <= 1 for k in ("predator_boundary_fraction", "prey_boundary_fraction")
        ),
        "Visibility/boundary measurements differ.",
    )
    need(
        health["all_finite"] is True
        and health["action_scalars"] == 600
        and 0 <= health["saturation_fraction"] <= 1
        and 0 <= health["max_abs_action"] <= 1
        and health["max_abs_loc"] >= 0
        and 0 < health["min_scale"] <= health["max_scale"],
        "Action/distribution health failed.",
    )
    senders = (
        (0 if arm == "severed" else 2 if arm.startswith("suppress_") else 3)
        if model == "pcp_broadcast_k3"
        else 0
    )
    width = 32 if model in ("pcp_broadcast_k0", "pcp_broadcast_k3") else 0
    expected = {
        "sender_emissions": 100 * senders,
        "edge_deliveries": 200 * senders,
        "sender_payload_bits": 100 * senders * width * 32,
        "edge_delivery_payload_bits": 200 * senders * width * 32,
        "transitions": 100,
        "packet_scalars": width,
        "bits_per_scalar": 32,
        "cost_model": "one_float32_broadcast_packet_per_available_sender; two_receivers",
        "network_headers_counted": False,
        "measured_network_traffic": False,
    }
    need(costs == expected, "Observed packet payload differs from declared intervention.")
    stats = episode["module_stats"]
    need(
        all(finite(v) for v in stats.values())
        and stats.get("messages_per_step") == senders
        and stats.get("realized_sender_bits_per_step") == senders * width * 32
        and stats.get("realized_bits_per_step") == 2 * senders * width * 32,
        "Module packet accounting differs from observations.",
    )


def aggregate_bank(episodes):
    def mean(values):
        return math.fsum(values) / len(episodes)

    result = {key: mean([r["metrics"][key] for r in episodes]) for key in DOMAIN}
    contacted = [
        r["metrics"]["first_contact_step_or_horizon"]
        for r in episodes
        if r["metrics"]["first_contact_observed"]
    ]
    result["first_contact_conditional_mean"] = (
        math.fsum(contacted) / len(contacted) if contacted else None
    )
    result.update(
        episodes=len(episodes),
        transitions=100 * len(episodes),
        action_scalars=600 * len(episodes),
        action_all_finite=True,
        action_saturation_fraction=mean(
            [r["action_health"]["saturation_fraction"] for r in episodes]
        ),
    )
    for key in ("max_abs_action", "max_abs_loc", "max_scale", "min_scale"):
        operation = min if key == "min_scale" else max
        result[key] = operation(r["action_health"][key] for r in episodes)
    for prefix in ("sender", "edge_delivery"):
        values = [r["costs"][f"{prefix}_payload_bits"] for r in episodes]
        result[f"{prefix}_payload_bits_per_step"] = mean(values) / 100
        result[f"total_{prefix}_payload_bits"] = sum(values)
    return result


def verify_bank(bank, row, reference=None):
    arms = ARMS if row["model"] == "pcp_broadcast_k3" else ARMS[:2]
    need(
        set(bank["interventions"]) == set(arms) and bank["episode_seeds"] == EPISODES,
        "Missing, duplicate or unexpected complete arm bank.",
    )
    live = bank["interventions"]["live"]
    if reference is not None:
        need(len(reference) == 256, "Missing cross-policy episode reference.")
    aggregates = {}
    for arm in arms:
        episodes = bank["interventions"][arm]
        need(len(episodes) == 256, "An arm must contain all 256 episodes.")
        for index, (episode, seed) in enumerate(zip(episodes, EPISODES, strict=True)):
            verify_episode(episode, seed, row["model"], arm)
            need(
                all(episode[k] == live[index][k] for k in PAIR_KEYS),
                "Within-policy physical/exogenous episode pairing differs.",
            )
            if reference is not None:
                need(
                    all(episode[k] == reference[index][k] for k in PAIR_KEYS),
                    "Cross-policy/visibility physical episode pairing differs.",
                )
            if row["model"] != "pcp_broadcast_k3" and arm == "severed":
                need(
                    episode["metrics"] == live[index]["metrics"]
                    and episode["action_health"] == live[index]["action_health"],
                    "Local control failed its exact null intervention.",
                )
        aggregates[arm] = aggregate_bank(episodes)
    close(row["arms"], aggregates, f"{row['run_id']}/policy means")
    return aggregates, [{k: e[k] for k in PAIR_KEYS} for e in live]


def verify_packets(path, bank, inputs):
    inputs.pin(path, bank["packet_file_sha256"])
    import torch

    packet_record = torch.load(path, map_location="cpu", weights_only=True)
    packet = packet_record["packets"]
    need(
        packet_record["episode_seeds"] == EPISODES
        and isinstance(packet, torch.Tensor)
        and tuple(packet.shape) == (256, 100, 1, 3, 32)
        and packet.dtype == torch.float32
        and bool(torch.isfinite(packet).all()),
        "Frozen live packet tensor differs.",
    )

    def tensor_digest(tensor):
        tensor = tensor.detach().cpu().contiguous()
        hasher = hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode())
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return hasher.hexdigest()

    need(tensor_digest(packet) == bank["live_packet_bank_sha256"], "Live tensor hash differs.")
    for index, episode in enumerate(bank["interventions"]["shuffle"]):
        donor = (index + 1) % 256
        need(
            episode["donor_episode_seed"] == EPISODES[donor]
            and episode["donor_packet_sha256"] == tensor_digest(packet[donor]),
            "Cyclic donor identity or tensor hash differs.",
        )
    return {
        "shape": list(packet.shape),
        "dtype": str(packet.dtype),
        "donors_verified": 256,
        "packet_sha256": inputs.files[str(path)]["sha256"],
    }


def evaluation_inputs(root, row, plan, manifest, queue_dir, metadata, native, inputs):
    evaluation = queue_dir / "evaluations" / row["run_id"]
    need(Path(row["evaluation_directory"]) == evaluation, "Held-out directory differs.")
    names = {
        "heldout_evaluation.json",
        "retained_checkpoints.json",
        "training_validation.json",
        "action_audit.json",
        "domain_audit.json",
        "status.json",
    }
    if row["model"] == "pcp_broadcast_k3":
        names.add("live_packet_bank.pt")
    need(set(inputs.tree(evaluation)) == names, "Held-out artifact inventory differs.")
    bank = inputs.read(evaluation / "heldout_evaluation.json")
    need(
        inputs.read(evaluation / "status.json")["status"] == "completed",
        "Held-out evaluation status is incomplete.",
    )
    expected = {
        "schema_version": 1,
        "evaluation_protocol": "pcp_budget_evaluation_v1",
        "run_id": row["run_id"],
        "training_seed": row["seed"],
        "model": row["model"],
        "visibility": row["visibility"],
        "group": "adversary",
        "exploration": "DETERMINISTIC",
        "deadline": 50,
        "protocol_sha256": PROTOCOL,
        "plan_sha256": PLAN,
        "inputs_unchanged": True,
        "shuffle_definition": "frozen_live_packets_from_cyclic_next_episode_at_same_step",
        "shuffle_distribution_shift": (
            "donor live trajectories substituted into receiver closed-loop trajectories"
        ),
    }
    need(all(bank.get(k) == v for k, v in expected.items()), "Held-out protocol/identity differs.")
    validation = inputs.read(
        evaluation / "training_validation.json", bank["training_validation_sha256"]
    )
    retained = inputs.read(
        evaluation / "retained_checkpoints.json", bank["retained_checkpoints_sha256"]
    )
    need(
        retained == native
        and retained["schema_version"] == 1
        and retained["run_id"] == row["run_id"]
        and retained["final_native_actor_matches_managed"] is True,
        "Current retained inventory differs from archived validated headers.",
    )
    need(
        len(retained["checkpoints"]) == 5
        and sorted(r["frames"] for r in retained["checkpoints"])
        == list(range(120000, 600001, 120000)),
        "Required five native checkpoints differ.",
    )
    run = Path(row["run_directory"])
    native_paths = [Path(r["path"]) for r in retained["checkpoints"]]
    replay_paths = [Path(r["path"]) for r in retained["replay_diagnostics"]]
    need(len(set(native_paths)) == len(native_paths)
         and set(native_paths) == set(run.glob("benchmarl/*/checkpoints/checkpoint_*.pt"))
         and len(set(replay_paths)) == len(replay_paths)
         and set(replay_paths) == set(run.glob("benchmarl/*/replay_diagnostics/*.pt")),
         "Retained native/replay file inventory differs from raw run.")
    paths = [
        manifest,
        root / plan["protocol"]["path"],
        root / "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
        root / "scripts/evaluate_pcp_budget.py",
        *(run / name for name in RUN_INPUTS),
    ]
    for checkpoint in [*retained["checkpoints"], *retained["replay_diagnostics"]]:
        path = Path(checkpoint["path"])
        need(path.is_relative_to(run), "Native/replay checkpoint lies outside its run.")
        if "frames" in checkpoint:
            need(
                path.name == f"checkpoint_{checkpoint['frames']}.pt"
                and checkpoint["iterations"] == checkpoint["frames"] // 6000,
                "Retained native checkpoint header differs.",
            )
        inputs.pin(path, checkpoint["sha256"])
        paths.append(path)
    expected_inputs = {str(path): inputs.pin(path) for path in paths}
    need(bank["input_artifact_sha256"] == expected_inputs, "Held-out input binding differs.")
    need(
        validation["validation_passed"] is True
        and validation["issues"] == []
        and validation["run_id"] == row["run_id"]
        and validation["seed"] == row["seed"]
        and set(validation["artifacts"]) == set(RUN_INPUTS)
        and validation["performance"] == row["performance"],
        "Training validation binding differs.",
    )
    for name, value in validation["artifacts"].items():
        inputs.pin(run / name, value)
    for name in ("action", "domain"):
        need(
            canonical(inputs.read(evaluation / f"{name}_audit.json"))
            == validation[f"{name}_audit_sha256"],
            "Final action/domain audit binding differs.",
        )
    provenance = bank["analysis_provenance"]
    binding = {
        "source_run": str(run),
        "source_sha256": SOURCE,
        "checkpoint_sha256": retained["managed_final_policy_sha256"],
        "resolved_config_sha256": expected_inputs[str(run / "resolved_config.yaml")],
        "scientific_config_sha256": metadata["scientific_config_sha256"],
        "versions": metadata["versions"],
        "task_runtime_contract": metadata["task_runtime_contract"],
        "allow_runtime_mismatch": False,
        "compatibility_mismatches": {},
    }
    need(all(provenance.get(k) == v for k, v in binding.items()), "Held-out actor/runtime differs.")
    review = inputs.read(queue_dir / f"{row['run_id']}.review.json")
    need(
        review["checks_passed"] is True
        and review["input_artifact_sha256"].get(str(evaluation / "heldout_evaluation.json"))
        == inputs.files[str(evaluation / "heldout_evaluation.json")]["sha256"],
        "Coordinator strict review does not bind this bank.",
    )
    for path, value in review["input_artifact_sha256"].items():
        inputs.pin(Path(path), value)
    return bank


def contrast_terms():
    result = {}
    for visibility in VISIBILITIES:
        live = (1, "pcp_broadcast_k3", visibility, "live")
        for model in ("pcp_local_capacity", "pcp_broadcast_k0", "pcp_comm_identity"):
            result[f"broadcast_k3_minus_{model}/{visibility}", "contact_by_deadline"] = [
                live,
                (-1, model, visibility, "live"),
            ]
        for arm in ARMS[1:]:
            for metric in ("contact_by_deadline", "return"):
                result[f"broadcast_k3_live_minus_{arm}/{visibility}", metric] = [
                    live,
                    (-1, "pcp_broadcast_k3", visibility, arm),
                ]
    result[
        "capacity_controlled_visibility_interaction_radius1_minus_global", "contact_by_deadline"
    ] = [
        (1, "pcp_broadcast_k3", "radius1", "live"),
        (-1, "pcp_local_capacity", "radius1", "live"),
        (-1, "pcp_broadcast_k3", "global", "live"),
        (1, "pcp_local_capacity", "global", "live"),
    ]
    return result


def estimands(policies):
    need(
        set(policies) == set(itertools.product(MODELS, VISIBILITIES, SEEDS)),
        "No statistics may be computed from a reduced cohort.",
    )
    differences = {}
    for key, terms in contrast_terms().items():
        differences[key] = [
            sum(c * policies[m, v, seed][a][key[1]] for c, m, v, a in terms) for seed in SEEDS
        ]
    budgets = {
        (v, k): [
            policies[f"pcp_broadcast_k{k}", v, seed]["live"]["contact_by_deadline"]
            for seed in SEEDS
        ]
        for v, k in itertools.product(VISIBILITIES, (0, 3))
    }
    return differences, budgets


def bootstrap_interval(values):
    need(
        len(values) == 5 and all(finite(v) and type(v) is not bool for v in values),
        "Bootstrap needs five finite training-seed values.",
    )
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(73191)
    sample_ids = generator.integers(5, size=(10000, 5))
    means = array[sample_ids].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "training_seeds": 5,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 73191,
        "unit": "training_seed",
        "method": "paired_seed_percentile_bootstrap",
    }


def compare_interval(actual, expected, label):
    need(isinstance(actual, dict) and set(expected) <= set(actual), f"Missing interval: {label}")
    close({k: actual[k] for k in expected}, expected, label)


def review_statistics(report, policies):
    differences, budgets = estimands(policies)
    records = {(r["contrast"], r["metric"]): r for r in report["contrasts"]}
    need(
        len(report["contrasts"]) == 27 and set(records) == set(differences),
        "Missing/duplicated contrast and metric identity.",
    )
    intervals = {}
    for key, values in differences.items():
        record = records[key]
        expected_terms = [
            {"coefficient": c, "model": m, "visibility": v, "arm": a}
            for c, m, v, a in contrast_terms()[key]
        ]
        need(
            record["status"] == "complete"
            and record["missing_or_invalid_seeds"] == []
            and record["terms"] == expected_terms
            and record["per_seed"]
            == [{"seed": s, "value": v} for s, v in zip(SEEDS, values, strict=True)],
            f"Paired seed effect or contrast sign differs: {key}",
        )
        interval = bootstrap_interval(values)
        compare_interval(record["interval"], interval, str(key))
        intervals[key] = interval
    frontiers = {r["visibility"]: r for r in report["budget_frontiers"]}
    need(
        len(report["budget_frontiers"]) == 2 and set(frontiers) == set(VISIBILITIES),
        "Missing/duplicated frontier.",
    )
    decisions = {}
    for visibility in VISIBILITIES:
        frontier = frontiers[visibility]
        need(
            frontier["target"] == 0.8
            and frontier["deadline"] == 50
            and [r["sender_budget"] for r in frontier["budgets"]] == [0, 3],
            "Frozen frontier rule differs.",
        )
        qualified = []
        for record in frontier["budgets"]:
            k = record["sender_budget"]
            values = budgets[visibility, k]
            interval = bootstrap_interval(values)
            compare_interval(record["interval"], interval, f"{visibility}/K{k}")
            need(
                record["per_seed"]
                == [{"seed": s, "value": v} for s, v in zip(SEEDS, values, strict=True)]
                and record["qualifies"] is (interval["ci95_low"] >= 0.8),
                "Budget values or lower-bound qualification differs.",
            )
            if interval["ci95_low"] >= 0.8:
                qualified.append(k)
        choice = min(qualified) if qualified else "not reached"
        need(
            type(frontier["smallest_tested_budget"]) is type(choice)
            and frontier["smallest_tested_budget"] == choice,
            "Smallest tested budget differs.",
        )
        decisions[visibility] = choice
    conditions = {(r["model"], r["visibility"]): r for r in report["conditions"]}
    need(
        len(report["conditions"]) == 8
        and set(conditions) == set(itertools.product(MODELS, VISIBILITIES)),
        "Missing/duplicated condition.",
    )
    for (model, visibility), condition in conditions.items():
        need(
            condition["status"] == "complete" and condition["valid_seeds"] == list(SEEDS),
            "Condition omits training seeds.",
        )
        values_by_seed = [policies[model, visibility, seed] for seed in SEEDS]
        arm_means = {}
        for arm in values_by_seed[0]:
            arm_means[arm] = {}
            for metric in values_by_seed[0][arm]:
                values = [row[arm][metric] for row in values_by_seed]
                if all(finite(value) for value in values):
                    arm_means[arm][metric] = math.fsum(values) / 5
        live_means = {}
        for metric in values_by_seed[0]["live"]:
            values = [row["live"][metric] for row in values_by_seed]
            if all(finite(value) and type(value) is not bool for value in values):
                live_means[metric] = math.fsum(values) / 5
        close(condition["interventions"], arm_means, f"{model}/{visibility}/all-arm means")
        close(condition["means"], live_means, f"{model}/{visibility}/live means")
        for metric in CONDITION_CIS:
            values = [policies[model, visibility, seed]["live"][metric] for seed in SEEDS]
            expected = bootstrap_interval(values)
            compare_interval(
                condition["intervals"][metric], expected, f"{model}/{visibility}/{metric}"
            )
            close(
                condition["means"][metric], expected["mean"], f"{model}/{visibility}/mean/{metric}"
            )
    return {
        "paired_contrasts_verified": 27,
        "condition_intervals_verified": 56,
        "budget_intervals_verified": 4,
        "smallest_tested_budgets": decisions,
        "primary_contrast": {
            "name": PRIMARY,
            "metric": "contact_by_deadline",
            **intervals[PRIMARY, "contact_by_deadline"],
        },
    }


def audit(root, analysis_dir, queue_path, destination):
    root, analysis_dir, queue_path, destination = (
        Path(p).absolute() for p in (root, analysis_dir, queue_path, destination)
    )
    need(
        not destination.exists() and not destination.is_symlink(),
        "Preserve any prior audit attempt.",
    )
    need(
        not destination.is_relative_to(analysis_dir)
        and not destination.is_relative_to(root / "runs")
        and not destination.is_relative_to(queue_path.parent / "evaluations")
        and not destination.is_relative_to(queue_path.parent / "run_archives"),
        "Audit receipt must be outside protected scientific evidence.",
    )
    inputs = Inputs()
    inputs.pin(Path(__file__).resolve())
    queue = inputs.read(queue_path)
    report = inputs.read(analysis_dir / "analysis_report.json")
    closed_cohort(report, queue)
    package_files = package_snapshot(root, inputs)
    for path, value in FROZEN_FILES.items():
        inputs.pin(root / path, value)
    need(
        set(inputs.tree(analysis_dir)) == OUTPUTS | {"provenance.json"},
        "Analyzer output directory has missing or extra artifacts.",
    )
    provenance = inputs.read(analysis_dir / "provenance.json")
    need(
        provenance["schema_version"] == 1
        and provenance["input_artifact_hashes"] == report["artifact_hashes"]
        and set(provenance["output_artifact_hashes"]) == OUTPUTS
        and provenance["script_sha256"] == FROZEN_FILES["scripts/analyze_pcp_budget.py"]
        and provenance["training_source_sha256"] == provenance["analysis_source_sha256"] == SOURCE
        and provenance["no_training_or_policy_rollout"] is True,
        "Analyzer provenance differs.",
    )
    for name, value in provenance["output_artifact_hashes"].items():
        inputs.pin(analysis_dir / name, value)
    for name, value in report["artifact_hashes"].items():
        inputs.pin(Path(name), value)
    manifest = Path(queue["manifest"])
    need(
        report["artifact_hashes"].get(str(manifest)) == queue["manifest_sha256"],
        "Queue manifest is not bound by this analyzer report.",
    )
    plans = manifest_rows(manifest, report, inputs)
    policies, archive_reviews, packet_reviews, reference = {}, [], [], None
    for row in report["rows"]:
        q, plan = queue["rows"][row["run_id"]], plans[row["run_id"]]
        metadata, native, archive_review = verify_run_archive(
            root, row, q, plan, manifest, queue_path.parent, package_files, inputs
        )
        bank = evaluation_inputs(
            root, row, plan, manifest, queue_path.parent, metadata, native, inputs
        )
        values, identities = verify_bank(bank, row, reference)
        reference = identities if reference is None else reference
        policies[row["model"], row["visibility"], row["seed"]] = values
        if row["model"] == "pcp_broadcast_k3":
            packet_review = verify_packets(
                Path(row["evaluation_directory"]) / "live_packet_bank.pt", bank, inputs
            )
            packet_reviews.append({"run_id": row["run_id"], **packet_review})
        else:
            need(
                bank.get("packet_file_sha256") is None
                and bank.get("live_packet_bank_sha256") is None,
                "A zero-communication control unexpectedly declares a packet bank.",
            )
        archive_reviews.append(archive_review)
    statistics = review_statistics(report, policies)
    need(package_snapshot(root, inputs) == package_files, "Package inventory changed during audit.")
    inputs.finish()
    result = {
        "schema_version": 1,
        "checks_passed": True,
        "reviewed_utc": datetime.now(UTC).isoformat(),
        "status": "complete_evidence_independently_reviewed",
        "approval_decision": "not_issued",
        "source_sha256": SOURCE,
        "protocol_sha256": PROTOCOL,
        "plan_sha256": PLAN,
        "policies": 40,
        "heldout_episodes": 30720,
        "heldout_transitions": 3072000,
        "exact_local_null_policies": 30,
        "cross_policy_episode_pairing_verified": True,
        "source_before_after_equal": True,
        "all_archive_members_readback_verified": True,
        "archives": archive_reviews,
        "packet_banks": packet_reviews,
        "statistics": statistics,
        "input_artifact_sha256": {p: r["sha256"] for p, r in inputs.files.items()},
        "input_artifact_records": inputs.files,
        "exact_file_inventories": inputs.trees,
        "driver_sha256": inputs.files[str(Path(__file__).resolve())]["sha256"],
        "comparison_rounding_tolerance": {"absolute": 1e-12, "relative": 1e-12},
        "no_training_or_policy_rollout": True,
        "no_checkpoint_deserialization": True,
        "scope": "Raw held-out metrics, exact pairing/nulls/payload, packet tensors/donors, "
        "archive members/current sources, and independent five-seed NumPy bootstrap. "
        "Native checkpoint header/content validation and final actor equality are bound "
        "through retained hashed inventories, not reconstructed here. "
        "No scientific approval, retuning, convergence or untested bandwidth claim.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordinary(destination.parent, directory=True)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--queue-status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.root, args.analysis_dir, args.queue_status, args.out)
    print(
        json.dumps(
            {
                "checks_passed": result["checks_passed"],
                "path": str(args.out),
                "sha256": digest(args.out.absolute()),
                "statistics": result["statistics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
