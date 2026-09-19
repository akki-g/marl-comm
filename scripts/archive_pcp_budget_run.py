"""Preserve one completed D4 run and its closed worker log in a verified gzip tar.

The destination is a new directory. A completion manifest is published only
after a full archive read-back and source-immutability check. Original evidence
is never deleted. Failed partial archives remain available for inspection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import itertools
import json
import os
import stat
import tarfile
from contextlib import suppress
from pathlib import Path, PurePosixPath

import torch

from commstudy.experiments.protocols import content_sha256, load_protocol
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.sweeps import read_manifest, validate_plan


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e"
PROTOCOL_ID = "pcp_visibility_budget_cpu_v1"
PROTOCOL_SHA = "ec7c109413625f270b79bffc1753adf5b7691d008eb90a4297d844b020898b0a"
INVENTORY_HELPER_SHA = "223d03a1ac5368d7d9ed6db032ac3eaa424e979dae27e42b2b19fbda36768594"
MODELS = ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0", "pcp_broadcast_k3")
EMBEDDED_MANIFEST = "ARCHIVE_MANIFEST.json"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(document):
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def regular_files(root):
    """Never follow links or block on a special file while collecting evidence."""
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "Run must be an ordinary directory")
    result = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        need(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f"Non-regular evidence: {path}")
        if stat.S_ISREG(mode):
            result[path.relative_to(root).as_posix()] = path
    need(bool(result), "Empty run directory")
    return result


def inventory_reader():
    path = ROOT / "scripts/evaluate_pcp_budget.py"
    need(sha256(path) == INVENTORY_HELPER_SHA, "Reviewed native inventory helper changed")
    spec = importlib.util.spec_from_file_location("pcp_budget_archive_native_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.retained_checkpoint_inventory


def validate_inputs(manifest, run_id, worker_log):
    plans = read_manifest(manifest)
    expected = set(itertools.product(MODELS, ("radius1", "global"), range(20, 25)))
    need(
        len(plans) == 40
        and {(p.model, p.ablation_value, p.seed) for p in plans} == expected
        and len({p.run_id for p in plans}) == 40
        and len({(p.output_root.resolve(), p.suite_id) for p in plans}) == 1,
        "Archive requires the complete unique forty-row D4 allocation",
    )
    matches = [p for p in plans if p.run_id == run_id]
    need(len(matches) == 1, "Select exactly one declared run ID")
    plan = matches[0]
    need(
        plan.protocol is not None
        and plan.protocol.get("source_sha256") == SOURCE
        and plan.protocol.get("sha256") == PROTOCOL_SHA
        and plan.protocol.get("stage") == "comparison"
        and plan.ablation == "visibility"
        and plan.max_n_frames == 600000
        and plan.attempt == 0
        and plan.retry_of is None,
        "Run differs from the frozen D4 source/protocol/original attempt",
    )
    need(source_fingerprint(ROOT) == SOURCE, "Package source changed before archival")
    validate_plan(plan, config_root=ROOT / "configs", repo_root=ROOT, check_runtime=True)
    run = (plan.output_root / plan.suite_id / plan.run_id).resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    status = json.loads((run / "status.json").read_text())
    need(status.get("status") == "completed", "Managed status is not completed")
    expected_metadata = {
        "run_id": plan.run_id,
        "suite_id": plan.suite_id,
        "task": "vmas_predator_capture_prey",
        "algorithm": "mappo",
        "model": plan.model,
        "seed": plan.seed,
        "ablation": "visibility",
        "ablation_value": plan.ablation_value,
        "status": "completed",
        "frames": 600000,
        "iterations": 100,
        "execution_purpose": "scientific",
        "source_sha256": SOURCE,
        "scientific_config_sha256": plan.scientific_hash,
    }
    for key, value in expected_metadata.items():
        need(
            type(metadata.get(key)) is type(value) and metadata[key] == value,
            f"Managed run identity differs: {key}",
        )
    need(
        metadata["protocol"]["protocol_id"] == PROTOCOL_ID
        and metadata["protocol"]["protocol_sha256"] == PROTOCOL_SHA
        and metadata["protocol"]["source_sha256"] == SOURCE
        and metadata["protocol"]["stage"] == "comparison"
        and metadata["protocol"]["factors"]
        == {"model": plan.model, "visibility": plan.ablation_value},
        "Managed protocol binding differs",
    )
    need(
        metadata["task_runtime_contract"] == plan.model_contract["task_contract"]
        and metadata["parameters"] == plan.model_contract["parameter_counts"],
        "Task or actor/critic capacity contract differs",
    )
    protocol_path = ROOT / plan.protocol["path"]
    protocol = load_protocol(protocol_path)
    need(
        protocol.get("status") == "frozen" and content_sha256(protocol) == PROTOCOL_SHA,
        "Frozen protocol contents changed",
    )
    need(
        metadata["versions"]
        == {"python": protocol["runtime"]["python"], **protocol["runtime"]["packages"]},
        "Saved versions differ",
    )
    need(
        all(
            metadata["runtime"].get(key) == "cpu"
            for key in ("train_device", "sampling_device", "buffer_device")
        )
        and metadata["runtime"].get("torch_num_threads") == 1,
        "Saved CPU runtime differs",
    )
    worker_log = Path(worker_log)
    need(
        worker_log.name == f"{run_id}.log" and worker_log.is_file() and not worker_log.is_symlink(),
        "Require the run's ordinary closed queue worker log",
    )
    marker = f"[sweep] {run_id} -> completed".encode()
    need(marker in worker_log.read_bytes(), "Worker log lacks its completed run marker")
    native = inventory_reader()(run)
    managed = torch.load(
        run / "checkpoints/policy_state.pt", map_location="cpu", weights_only=False
    )
    for key in ("run_id", "suite_id", "task", "algorithm", "model", "seed", "frames"):
        value = expected_metadata[key]
        need(
            type(managed.get(key)) is type(value) and managed[key] == value,
            f"Final policy checkpoint identity differs: {key}",
        )
    need(
        type(managed.get("schema_version")) is int and managed["schema_version"] == 1,
        "Final policy schema differs",
    )
    return plan, run, metadata, native, protocol_path


def file_record(path):
    path = Path(path)
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode), f"Archive input is not an ordinary file: {path}")
    return {
        "source_path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "mode": stat.S_IMODE(mode),
    }


def write_tar(path, files, document):
    """Stream the input bytes; no uncompressed archive or whole-file buffers."""
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name, record in document["files"].items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = record["bytes"], record["mode"], 0
                    with files[name].open("rb") as stream:
                        tar.addfile(info, stream)
                payload = json_bytes(document)
                info = tarfile.TarInfo(EMBEDDED_MANIFEST)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())


def verify_archive(path):
    """Verify every archived byte using only the archive, without extracting it."""
    actual, embedded = {}, None
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name = member.name
            pure = PurePosixPath(name)
            need(
                member.isfile()
                and not pure.is_absolute()
                and ".." not in pure.parts
                and name not in actual,
                f"Invalid or duplicate archive member: {name}",
            )
            stream = archive.extractfile(member)
            if name == EMBEDDED_MANIFEST:
                need(embedded is None, "Duplicate embedded manifest")
                payload = stream.read()
                embedded = json.loads(payload)
                digest = hashlib.sha256(payload).hexdigest()
            else:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            actual[name] = {"sha256": digest, "bytes": member.size, "mode": member.mode}
    need(
        isinstance(embedded, dict)
        and type(embedded.get("schema_version")) is int
        and embedded["schema_version"] == 1,
        "Missing embedded archive manifest",
    )
    need(
        set(actual) == set(embedded["files"]) | {EMBEDDED_MANIFEST},
        "Archive contents differ from the embedded file inventory",
    )
    for name, expected in embedded["files"].items():
        need(
            all(actual[name][key] == expected[key] for key in ("sha256", "bytes", "mode")),
            f"Archive read-back differs: {name}",
        )
    # Force gzip to consume its trailer too: a truncated/corrupt footer must fail.
    with gzip.open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    return embedded


def publish_json(path, document):
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)  # Exclusive publication, including concurrent destination creation.
    with suppress(OSError):
        pending.unlink()  # Cleanup cannot invalidate an already-published verified file.


def archive_run(manifest, run_id, worker_log, destination):
    manifest, worker_log = Path(manifest), Path(worker_log)
    destination = Path(destination)
    need(not destination.exists(), "Refusing to overwrite an existing archive destination")
    matches = [p for p in read_manifest(manifest) if p.run_id == run_id]
    need(len(matches) == 1, "Select exactly one declared run ID")
    candidate = (matches[0].output_root / matches[0].suite_id / run_id).resolve()
    need(
        not destination.resolve().is_relative_to(candidate),
        "Archive must be outside its source run",
    )
    before_run_files = regular_files(candidate)
    before_run = {name: file_record(path) for name, path in before_run_files.items()}
    before_worker, before_manifest = file_record(worker_log), file_record(manifest)
    # Reserve first; invalid input leaves a labelled failed attempt, never a success marker.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        plan, run, metadata, native, protocol_path = validate_inputs(manifest, run_id, worker_log)
        need(
            not destination.resolve().is_relative_to(run), "Archive must be outside its source run"
        )
        need(run == candidate, "Manifest run path changed during validation")
        run_files = regular_files(run)
        need(run_files == before_run_files, "Run file set changed during validation")
        files = {f"run/{name}": path for name, path in run_files.items()}
        files.update(
            {
                "worker/stdout.log": worker_log,
                "provenance/worker_manifest.csv": manifest,
                "provenance/protocol.yaml": protocol_path,
                "provenance/plan.md": ROOT / "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
                "provenance/archive_helper.py": Path(__file__),
                "provenance/inventory_helper.py": ROOT / "scripts/evaluate_pcp_budget.py",
                "provenance/source_freeze.json": ROOT
                / "results/pcp_visibility_budget_cpu_v1/package_source_freeze.json",
                "source/pyproject.toml": ROOT / "pyproject.toml",
                "source/uv.lock": ROOT / "uv.lock",
            }
        )
        approval = ROOT / plan.protocol["approval"]
        need(
            content_sha256(json.loads(approval.read_text()))
            == metadata["protocol"]["approval_sha256"],
            "Saved approval artifact differs",
        )
        files["provenance/launch_approval.json"] = approval
        for path in sorted((ROOT / "src/commstudy").rglob("*.py")):
            files[f"source/{path.relative_to(ROOT).as_posix()}"] = path
        records = {name: file_record(path) for name, path in sorted(files.items())}
        need(
            records["worker/stdout.log"] == before_worker
            and records["provenance/worker_manifest.csv"] == before_manifest,
            "Worker log or manifest changed during validation",
        )
        need(
            all(records[f"run/{name}"] == record for name, record in before_run.items()),
            "Run file changed during validation",
        )
        by_source = {record["source_path"]: record for record in records.values()}
        for checkpoint in [*native["checkpoints"], *native["replay_diagnostics"]]:
            need(
                by_source[str(Path(checkpoint["path"]).resolve())]["sha256"]
                == checkpoint["sha256"],
                "Native evidence changed after header validation",
            )
        need(
            records["run/checkpoints/policy_state.pt"]["sha256"]
            == native["managed_final_policy_sha256"],
            "Managed policy changed after native comparison",
        )
        document = {
            "schema_version": 1,
            "archive": "pcp_budget_completed_run_v1",
            "run_id": run_id,
            "suite_id": plan.suite_id,
            "source_sha256": SOURCE,
            "protocol_sha256": PROTOCOL_SHA,
            "frames": 600000,
            "iterations": 100,
            "model": plan.model,
            "seed": plan.seed,
            "visibility": plan.ablation_value,
            "run_file_count": len(run_files),
            "native_inventory": native,
            "files": records,
            "scope": "All completed run files and worker log; held-out audits archived separately",
            "claim": "Byte preservation and validated checkpoint identity, not scientific success",
        }
        partial = destination / "run.partial.tar.gz"
        write_tar(partial, files, document)
        need(verify_archive(partial) == document, "Embedded archive manifest changed")
        need(regular_files(run) == run_files, "Run file set changed during archival")
        need(source_fingerprint(ROOT) == SOURCE, "Package source changed during archival")
        for name, path in files.items():
            need(file_record(path) == records[name], f"Source changed during archival: {name}")
        archive_hash = sha256(partial)
        published = destination / "run.tar.gz"
        os.link(partial, published)
        with suppress(OSError):
            partial.unlink()
        receipt = {
            "schema_version": 1,
            "checks_passed": True,
            "run_id": run_id,
            "archive_file": published.name,
            "archive_sha256": archive_hash,
            "archive_bytes": published.stat().st_size,
            "uncompressed_file_bytes": sum(r["bytes"] for r in records.values()),
            "embedded_manifest_sha256": hashlib.sha256(json_bytes(document)).hexdigest(),
            "archived_files": len(records),
            "run_file_count": len(run_files),
            "source_before_after_equal": True,
            "all_archive_files_rehashed": True,
            "original_files_removed": False,
        }
        publish_json(destination / "archive_manifest.json", receipt)
        return receipt
    except Exception as error:
        publish_json(
            destination / "archive_failure.json",
            {
                "schema_version": 1,
                "checks_passed": False,
                "error": f"{type(error).__name__}: {error}",
                "partial_evidence_preserved": True,
                "original_files_removed": False,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    result = archive_run(args.manifest, args.run_id, args.worker_log, args.destination)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
