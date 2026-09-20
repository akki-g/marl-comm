"""Fresh fixed-MAPPO suite; immutable plans and fail-closed stage gates.

No training algorithm lives here. Scientific execution uses RunContext and the
managed BenchMARL runner, including its finite-value and PPO replay guards.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

from commstudy.experiments.bookkeeping import atomic_write_json, read_json, utc_now
from commstudy.experiments.config import (
    experiment_spec_from_dict,
    resolved_experiment_dict,
)

ROOT = Path(__file__).resolve().parents[3]
SUITE = "communication_mechanisms_v1"
METHODS = ("identity", "local_capacity", "broadcast", "gated", "attention", "graph_full")
CONDITIONS = {"pcp": ("radius1", "global"), "mapdn": ("case33",)}
BUDGETS = {"pcp": 600000, "mapdn": 131072}
CHECKPOINTS = {
    "pcp": [120000, 240000, 360000, 480000, 600000],
    "mapdn": [32768, 65536, 98304, 131072],
}
MODULES = {
    "identity": ("identity.IdentityComm", 0, True),
    "local_capacity": ("local_control.LocalCapacityComm", 0, True),
    "broadcast": ("broadcast.BroadcastComm", 32, False),
    "gated": ("gated.GatedComm", 32, False),
    "attention": ("attention.AttentionComm", 64, False),
    "graph_full": ("graph.GraphComm", 64, False),
}
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserved artifact already exists: {path}")
    atomic_write_json(path, value)


def inventory(root=ROOT):
    paths = []
    for name in (
        "src",
        "scripts",
        "configs",
        "vendor/mapdn-source",
        "slurm/communication_comparison_v1",
    ):
        paths.extend(
            p
            for p in (root / name).rglob("*")
            if p.is_file()
            and not any(x in p.parts for x in ("__pycache__", ".pytest_cache", ".git"))
            and not any(x.endswith(".egg-info") for x in p.parts)
            and p.suffix not in {".pyc", ".pyo"}
            and p.name != ".DS_Store"
        )
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists())
    return {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(paths)}


def registry():
    return {
        name: {
            "class_path": "commstudy.communication." + cls,
            "communicating": not null,
            "null_control": null,
            "packet_scalars": width,
            "bits_per_scalar": 32,
            "packet": "key+value" if width == 64 else "value" if width else "none",
            "context_keys": [] if null else ["mask", "sender_mask"],
        }
        for name, (cls, width, null) in MODULES.items()
    }


def load_suite(path):
    from omegaconf import OmegaConf

    doc = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    keys = {
        "schema_version",
        "suite_id",
        "methods",
        "seeds",
        "qualification_seeds",
        "smoke_seed",
        "include",
        "pcp_test_seeds",
        "pcp_development_seeds",
        "pcp_random_audit_seeds",
        "bootstrap_seed",
        "historical_roots",
    }
    if not isinstance(doc, dict) or set(doc) != keys:
        raise ValueError(f"Suite requires exactly {sorted(keys)}")
    if doc["schema_version"] != 1 or doc["suite_id"] != SUITE or doc["methods"] != list(METHODS):
        raise ValueError("Unknown suite schema/identity or unstable method order")
    namespaces = [
        doc[k]
        for k in (
            "seeds",
            "qualification_seeds",
            "pcp_test_seeds",
            "pcp_development_seeds",
            "pcp_random_audit_seeds",
        )
    ]
    namespaces += [[doc["smoke_seed"], doc["bootstrap_seed"]]]
    flat = [seed for values in namespaces for seed in values]
    if any(type(s) is not int or not 0 <= s < 2**32 for s in flat) or len(set(flat)) != len(flat):
        raise ValueError("Seed namespaces must contain unique disjoint uint32 integers")
    if (
        len(doc["seeds"]) < 5
        or len(doc["qualification_seeds"]) != 2
        or len(doc["pcp_test_seeds"]) != 256
    ):
        raise ValueError(
            "Require >=5 main seeds, two qualification seeds and 256 PCP test episodes"
        )
    if min(len(doc[k]) for k in ("pcp_development_seeds", "pcp_random_audit_seeds")) < 2:
        raise ValueError("Development and random audit banks need at least two episodes")
    if len(set(doc["include"])) != len(doc["include"]) or set(doc["include"]) - {
        "capacity",
        "mapdn_topology",
    }:
        raise ValueError("Unknown/duplicate supplement")
    if not isinstance(doc["historical_roots"], list):
        raise ValueError("historical_roots must be a list of explicit evidence directories")
    return doc


def make_spec(task, condition, method, seed, stage, root, data_path):
    from omegaconf import OmegaConf

    protocol = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/protocols" / f"{task}_mechanisms_v1.yaml"), resolve=True
    )
    if set(protocol) != {"schema_version", "protocol_id", "status", "base_spec", "methods"}:
        raise ValueError("Unknown suite protocol fields")
    if method not in {*METHODS, "local_capacity64", "graph_electrical", "graph_random"}:
        raise ValueError("Unsupported method")
    raw = deepcopy(protocol["base_spec"])
    core_method = (
        "local_capacity"
        if method == "local_capacity64"
        else ("graph_full" if method.startswith("graph_") else method)
    )
    if core_method not in METHODS or condition not in CONDITIONS[task]:
        raise ValueError("Undeclared condition/method")
    model = deepcopy(protocol["methods"][core_method])
    params = model["groups"]["adversary"]["params"] if task == "pcp" else model["params"]
    if method == "local_capacity64":
        params["comm_kwargs"]["message_dim"] = 64
    raw.update(model=f"{task}_mechanisms_{method}", seed=seed, model_config=model)
    if task == "pcp":
        raw["task_config"]["params"]["predator_sensing_radius"] = (
            1.0 if condition == "radius1" else None
        )
    else:
        raw["task_config"]["params"].update(
            data_path=str(data_path),
            manifest_path=str(root / "preparation/split_manifest.json"),
            normalization_path=str(root / "preparation/normalizer.json"),
        )
    if stage == "smoke":
        frames = 6000 if task == "pcp" else 512
        raw["experiment"].update(max_n_frames=frames, checkpoint_interval=frames, evaluation=False)
    return resolved_experiment_dict(experiment_spec_from_dict(raw))


def rows_for(doc, task, stage, root, data_path):
    rows = []
    seeds = (
        doc["seeds"]
        if stage == "main"
        else (doc["qualification_seeds"] if stage == "qualification" else [doc["smoke_seed"]])
    )
    methods = ("identity",) if stage == "qualification" else METHODS
    allocations = [
        ("core", condition, method) for condition in CONDITIONS[task] for method in methods
    ]
    if stage == "main":
        if "capacity" in doc["include"]:
            allocations += [("capacity", c, "local_capacity64") for c in CONDITIONS[task]]
        if task == "mapdn" and "mapdn_topology" in doc["include"]:
            allocations += [
                ("mapdn_topology", "case33", m) for m in ("graph_electrical", "graph_random")
            ]
    for supplement, condition, method in allocations:
        for seed in seeds:
            spec = make_spec(task, condition, method, seed, stage, root, data_path)
            row = {
                "index": len(rows),
                "task": task,
                "condition": condition,
                "method": method,
                "seed": seed,
                "stage": stage,
                "cohort": supplement,
                "run_id": f"{stage}_{condition}_{method}_{seed}",
                "frames": spec["experiment"]["max_n_frames"],
                "spec": spec,
                "scientific_config_sha256": portable_config_digest(spec),
            }
            row["sha256"] = digest(row)
            rows.append(row)
    return rows


def portable_config_digest(spec):
    """Separate scientific settings from placement; data/bank bytes bind at freeze."""
    raw = deepcopy(spec)
    for key in ("data_path", "manifest_path", "normalization_path"):
        raw["task_config"]["params"].pop(key, None)
    raw["experiment"].pop("save_folder", None)
    return digest(raw)


def plan(suite, out, *, data_path=None, include=None):
    root = Path(out).resolve()
    doc = load_suite(suite)
    if include is not None:
        doc["include"] = include
        if set(include) - {"capacity", "mapdn_topology"} or len(set(include)) != len(include):
            raise ValueError("Unknown/duplicate supplement")
    data_path = Path(data_path or ROOT / "data/case33_3min_final").resolve()
    if root == ROOT or ROOT in root.parents:
        raise ValueError("Study root must be outside the source snapshot")
    root.mkdir(parents=True, exist_ok=False)
    prep = root / "preparation"
    prep.mkdir()
    write_new(prep / "suite.json", doc)
    shutil.copyfile(suite, prep / "suite.yaml")
    write_new(prep / "method_registry.json", registry())
    write_new(prep / "source_inventory.json", inventory())
    manifest_files, counts = {}, {}
    for stage in ("main", "qualification", "smoke"):
        for task in CONDITIONS:
            rows = rows_for(doc, task, stage, root, data_path)
            name = f"{task}_{stage}_manifest.json"
            write_new(prep / name, rows)
            manifest_files[name] = file_sha(prep / name)
            fields = [k for k in rows[0] if k != "spec"]
            with (prep / name.replace(".json", ".csv")).open("x", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: r[k] for k in fields} for r in rows)
            counts[f"{task}_{stage}"] = len(rows)
    packet = {
        "suite_id": SUITE,
        "created_utc": utc_now(),
        "source_sha256": digest(inventory()),
        "data_path": str(data_path),
        "manifests": manifest_files,
        "counts": counts,
        "suite_sha256": file_sha(prep / "suite.json"),
        "method_registry_sha256": file_sha(prep / "method_registry.json"),
        "test_rollouts_performed": 0,
        "resources": "unmeasured; qualification required",
    }
    packet["sha256"] = digest(packet)
    write_new(prep / "plan.json", packet)
    return packet


def verify_plan(root, *, check_source=True):
    root = Path(root)
    prep = root / "preparation"
    packet = read_json(prep / "plan.json")
    if not packet or packet["sha256"] != digest({k: v for k, v in packet.items() if k != "sha256"}):
        raise ValueError("Plan missing or altered")
    expected = {
        **packet["manifests"],
        "suite.json": packet["suite_sha256"],
        "method_registry.json": packet["method_registry_sha256"],
    }
    for name, sha in expected.items():
        if file_sha(prep / name) != sha:
            raise ValueError(f"Plan input altered: {name}")
    if check_source and (
        digest(inventory()) != packet["source_sha256"]
        or read_json(prep / "source_inventory.json") != inventory()
    ):
        raise ValueError("Source changed after plan; use a new source snapshot and study root")
    return packet


def manifest(root, task, stage="main"):
    if task not in CONDITIONS or stage not in {"main", "qualification", "smoke"}:
        raise ValueError("Unknown task/stage")
    rows = read_json(Path(root) / "preparation" / f"{task}_{stage}_manifest.json")
    if not rows or any(
        r["index"] != i or r["sha256"] != digest({k: v for k, v in r.items() if k != "sha256"})
        for i, r in enumerate(rows)
    ):
        raise ValueError("Manifest is not dense or has altered rows")
    return rows


def select_row(root, task, stage, index):
    rows = manifest(root, task, stage)
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError(f"Index must be in [0,{len(rows) - 1}]")
    return rows[index]


def runtime(task):
    import torch
    import commstudy

    if Path(commstudy.__file__).resolve().parent != ROOT / "src/commstudy":
        raise ValueError("commstudy import is outside the source snapshot")
    if any(os.environ.get(k) != "1" for k in THREADS) or torch.get_num_threads() != 1:
        raise ValueError("Explicit one-thread environment required (OMP/MKL/VECLIB/NUMEXPR)")
    if torch.get_num_interop_threads() != 1:
        raise ValueError("PyTorch inter-op threads must be explicitly set to one")
    packages = ["torch", "torchrl", "tensordict", "benchmarl", "vmas", "numpy"]
    if task == "mapdn":
        packages += ["pandas", "pandapower", "pettingzoo", "gymnasium"]
        import mapdn

        if Path(mapdn.__file__).resolve().parent != ROOT / "vendor/mapdn-source/mapdn":
            raise ValueError("MAPDN must import the pinned vendored research adapter")
    modules = {p: importlib.import_module(p) for p in packages}
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": "cpu",
        "threads": 1,
        "interop_threads": 1,
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "package_artifacts": {
            p: file_sha(m.__file__) for p, m in modules.items() if getattr(m, "__file__", None)
        },
        "distribution_records": {
            p: digest(importlib.metadata.distribution(p).read_text("RECORD")) for p in packages
        },
        "torch_config": torch.__config__.show(),
    }


def gate(root, name):
    value = read_json(Path(root) / "gates" / f"{name}.json")
    if not value or value.get("status") != "passed":
        raise ValueError(f"Required gate pending/failed: {name}")
    return value


def verify_preflight(root, task, *, check_runtime=True):
    packet = verify_plan(root)
    record = gate(root, f"preflight_{task}")
    if record["plan_sha256"] != packet["sha256"]:
        raise ValueError("Preflight does not bind this plan")
    if file_sha(record["profile_path"]) != record["profile_sha256"]:
        raise ValueError(
            "Qualified profile changed; preserve it and use a separate operational profile"
        )
    if check_runtime and record["runtime"] != runtime(task):
        raise ValueError("Scientific runtime differs from preflight; no backend fallback")
    for name, sha in record["inputs"].items():
        if file_sha(Path(root) / name) != sha:
            raise ValueError(f"Preflight input changed: {name}")
    if task == "mapdn":
        from commstudy.tasks.torchrl.mapdn_data import load_split_manifest

        load_split_manifest(Path(root) / "preparation/split_manifest.json", packet["data_path"])
    return record


def effective_spec(root, row):
    raw = deepcopy(row["spec"])
    if row["method"] in {"graph_electrical", "graph_random"}:
        topology = read_json(Path(root) / "preparation/topologies.json")
        if not topology or topology.get("status") != "passed":
            raise ValueError("Topology supplement blocked: real mapping/matched mask unavailable")
        raw["model_config"]["params"]["comm_kwargs"]["fixed_mask"] = topology[row["method"]][
            str(row["seed"])
        ]
    return experiment_spec_from_dict(raw)


def contract_key(row):
    return (
        f"{row['condition']}:{row['method']}:{row['seed']}"
        if row["method"] == "graph_random"
        else f"{row['condition']}:{row['method']}"
    )


def verify_stage(root, task, stage):
    evidence = []
    for row in manifest(root, task, stage):
        result = latest_attempt(root, row)
        if not result or result["status"] != "completed" or not result.get("integrity_passed"):
            raise ValueError(f"{stage} incomplete: {row['run_id']}")
        verify_receipt(root, result)
        evidence.append(result)
    return evidence


def validate_launch(spec, binding, *, repo_root):
    from commstudy.experiments.protocols import ProtocolGateError

    if (
        set(binding) != {"kind", "run_root", "task", "stage", "index"}
        or Path(repo_root).resolve() != ROOT
    ):
        raise ProtocolGateError("Invalid new-suite launch binding")
    root = Path(binding["run_root"])
    row = select_row(root, binding["task"], binding["stage"], binding["index"])
    preflight = verify_preflight(root, row["task"])
    if asdict(spec) != asdict(effective_spec(root, row)):
        raise ProtocolGateError("Row settings differ from the sealed manifest")
    if row["stage"] == "qualification":
        verify_stage(root, row["task"], "smoke")
    elif row["stage"] == "main":
        verify_approval(root)
    return {
        "protocol_id": SUITE,
        "stage": row["stage"],
        "row_sha256": row["sha256"],
        "source_sha256": preflight["source_sha256"],
        "runtime": preflight["runtime"],
    }


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Exclusive ownership exists: {path}; never reclaimed by age") from exc
    try:
        write_new(
            path / "owner.json",
            {"pid": os.getpid(), "job": os.getenv("SLURM_JOB_ID"), "created_utc": utc_now()},
        )
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()


def latest_attempt(root, row, kind="runs"):
    base = Path(root) / kind / row["task"] / row["run_id"]
    attempts = sorted(base.glob("attempt_*/receipt.json"))
    pending = sorted(base.glob("attempt_*/start.json"))
    if pending and (not attempts or pending[-1].parent != attempts[-1].parent):
        return {"status": "interrupted_or_running", "path": str(pending[-1].parent)}
    if not attempts:
        return None
    receipt = read_json(attempts[-1])
    if receipt.get("row") != row:
        raise ValueError("Attempt receipt belongs to a different manifest row")
    return receipt


def verify_receipt(root, receipt):
    for path, sha in receipt.get("artifacts", {}).items():
        if file_sha(Path(root) / path) != sha:
            raise ValueError(f"Retained artifact differs: {path}")
    if receipt.get("plan_sha256") != verify_plan(root)["sha256"]:
        raise ValueError("Receipt belongs to a different plan")


def freeze(root):
    root = Path(root)
    packet = verify_plan(root)
    proofs = {}
    for task in CONDITIONS:
        preflight = verify_preflight(root, task, check_runtime=False)
        if not preflight["cluster_profile_verified"]:
            raise ValueError(
                "Target allocation preflight required; local smokes do not qualify cluster"
            )
        for stage in ("smoke", "qualification"):
            proofs[f"{task}_{stage}"] = verify_stage(root, task, stage)
    inputs = {
        p.relative_to(root).as_posix(): file_sha(p)
        for p in (root / "preparation").rglob("*")
        if p.is_file()
    }
    inputs.update(
        {
            p.relative_to(root).as_posix(): file_sha(p)
            for p in (root / "gates").glob("preflight_*.json")
        }
    )
    for results in proofs.values():
        for result in results:
            inputs.update(result["artifacts"])
            path = root / result["receipt_path"]
            inputs[result["receipt_path"]] = file_sha(path)
    approval = {
        "status": "passed",
        "plan_sha256": packet["sha256"],
        "inputs": inputs,
        "source_sha256": packet["source_sha256"],
        "created_utc": utc_now(),
        "budget_adequacy": "unresolved; fixed allocation, no convergence claim",
        "positive_communication_effect_required": False,
        "test_rollouts_performed": 0,
    }
    write_new(root / "gates/main_approval.json", approval)
    return approval


def verify_approval(root):
    packet = verify_plan(root)
    approval = gate(root, "main_approval")
    if approval["plan_sha256"] != packet["sha256"]:
        raise ValueError("Approval belongs to another plan")
    for path, sha in approval["inputs"].items():
        if file_sha(Path(root) / path) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    return approval
