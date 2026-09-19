"""Execute the bounded D4 allocation, preserving failures and stopping queue expansion."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.sweeps import read_manifest, validate_plan


MINIMUM_FREE_BYTES = 8 * 1024**3
ARCHIVE_HELPER_SHA256 = "36c9458f81a0477cbdc3cc31971ab3c92af19ea16d5a955c8fdca5fad3245e1f"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def storage_preflight(paths):
    checks = []
    for target in paths:
        probe = target.resolve()
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        checks.append({"target": str(target.resolve()), "checked_existing_path": str(probe),
                       "free_bytes": free, "required_free_bytes": MINIMUM_FREE_BYTES,
                       "passes": free >= MINIMUM_FREE_BYTES})
    return {"checks_passed": all(row["passes"] for row in checks), "paths": checks,
            "scope": "Initial forty-row allocation, including raw runs, compressed copies, "
                     "audits and operating-system headroom; checked on each output filesystem."}


def archive_completed(plan, manifest, archive_dir, worker_log, queue_dir):
    root = Path(__file__).resolve().parents[1]
    destination = archive_dir / plan.run_id
    if destination.exists():
        raise ValueError("Refusing to reuse an archive destination.")
    command = [sys.executable, str(root / "scripts/archive_pcp_budget_run.py"),
               "--manifest", str(manifest.resolve()), "--run-id", plan.run_id,
               "--destination", str(destination.resolve()),
               "--worker-log", str(worker_log.resolve())]
    with (queue_dir / f"{plan.run_id}.archive.log").open("x") as stream:
        process = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise ValueError(f"Run preservation failed with exit code {process.returncode}.")
    receipt_path = destination / "archive_manifest.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("Archive receipt must be an ordinary preserved file.")
    receipt = json.loads(receipt_path.read_text())
    expected = {"schema_version": 1, "checks_passed": True, "run_id": plan.run_id,
                "archive_file": "run.tar.gz", "source_before_after_equal": True,
                "all_archive_files_rehashed": True, "original_files_removed": False}
    for key, value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt[key] != value:
            raise ValueError(f"Archive receipt differs at {key}.")
    if (destination / "archive_failure.json").exists():
        raise ValueError("Archive contains a failure marker.")
    archive = destination / "run.tar.gz"
    if (not archive.is_file() or archive.is_symlink()
            or type(receipt.get("archive_bytes")) is not int
            or archive.stat().st_size != receipt["archive_bytes"]
            or sha256(archive) != receipt.get("archive_sha256")):
        raise ValueError("Published archive bytes do not match their verification receipt.")
    return {"destination": str(destination.resolve()), "receipt_sha256": sha256(receipt_path),
            "archive_sha256": receipt["archive_sha256"], "archive_bytes": receipt["archive_bytes"],
            "checks_passed": True}


def completed_metadata(plan):
    run = plan.output_root / plan.suite_id / plan.run_id
    metadata = json.loads((run / "metadata.json").read_text())
    expected = {
        "status": "completed", "frames": 600000, "iterations": 100,
        "run_id": plan.run_id, "suite_id": plan.suite_id,
        "task": "vmas_predator_capture_prey", "algorithm": "mappo",
        "model": plan.model, "seed": plan.seed, "ablation": "visibility",
        "ablation_value": plan.ablation_value, "execution_purpose": "scientific",
    }
    for key, value in expected.items():
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise ValueError(f"Completed worker metadata differs at {key}.")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if sha256(root / "scripts/archive_pcp_budget_run.py") != ARCHIVE_HELPER_SHA256:
        parser.error("The reviewed archive helper changed; revalidate it before launching.")
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    plans = read_manifest(args.manifest)
    expected = {
        (seed, visibility, model)
        for seed in range(20, 25)
        for visibility in ("radius1", "global")
        for model in (
            "pcp_comm_identity",
            "pcp_local_capacity",
            "pcp_broadcast_k0",
            "pcp_broadcast_k3",
        )
    }
    actual = {(p.seed, p.ablation_value, p.model) for p in plans}
    if len(plans) != 40 or actual != expected:
        parser.error("Manifest must contain exactly the prespecified forty D4 rows.")
    run_paths = [(p.output_root / p.suite_id / p.run_id).resolve() for p in plans]
    if len({p.run_id for p in plans}) != 40 or len(set(run_paths)) != 40:
        parser.error("Run IDs and resolved output directories must be unique.")
    if len({(p.output_root.resolve(), p.suite_id) for p in plans}) != 1:
        parser.error("The bounded allocation must belong to one managed suite.")
    archive_paths = [(args.archive_dir / p.run_id).resolve() for p in plans]
    if set(archive_paths) & set(run_paths) or any(
        args.archive_dir.resolve().is_relative_to(path)
        or args.out_dir.resolve().is_relative_to(path) for path in run_paths
    ):
        parser.error("Archive and queue evidence must remain outside every original run.")
    if args.archive_dir.exists() and not args.archive_dir.is_dir():
        parser.error("Archive root must be a directory.")
    if any(path.exists() for path in archive_paths):
        parser.error("Refusing to reuse any existing per-run archive destination.")
    for plan in plans:
        if any(Path(label).name != label or label in {"", ".", ".."}
               for label in (plan.run_id, plan.suite_id)):
            parser.error("Run and suite IDs must be single directory names.")
    for plan in plans:
        if plan.max_n_frames != 600000 or plan.ablation != "visibility":
            parser.error("Unexpected horizon or factor.")
        if plan.protocol is None or plan.protocol["stage"] != "comparison":
            parser.error("Every row requires a comparison-stage protocol binding.")
        validate_plan(plan, config_root=root / "configs", repo_root=root, check_runtime=True)
        run = plan.output_root / plan.suite_id / plan.run_id
        if run.exists():
            parser.error(f"Fresh allocation required; existing attempt must be preserved: {run}")
    if args.out_dir.exists():
        parser.error("Refusing to overwrite queue evidence.")
    storage = storage_preflight([plans[0].output_root, args.archive_dir, args.out_dir])
    if not storage["checks_passed"]:
        parser.error(f"Insufficient storage for the fixed allocation: {json.dumps(storage)}")
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != manifest_hash:
        parser.error("Manifest changed during preflight.")
    args.out_dir.mkdir(parents=True)
    snapshot = args.out_dir / "manifest.csv"
    with snapshot.open("xb") as stream:
        stream.write(manifest_bytes)
    operational_paths = [Path(__file__), root / "scripts/archive_pcp_budget_run.py",
                         root / "scripts/evaluate_pcp_budget.py",
                         root / "docs/PCP_VISIBILITY_BUDGET_PLAN.md"]
    operational_paths.extend(root / p.protocol["path"] for p in plans if "path" in p.protocol)
    operational_hashes = {str(path.resolve()): sha256(path) for path in operational_paths}
    # Adjacent rows make a paired seed/visibility block; no outcome-based ordering.
    model_order = {
        m: i
        for i, m in enumerate(
            ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0", "pcp_broadcast_k3")
        )
    }
    plans.sort(key=lambda p: (p.seed, p.ablation_value != "radius1", model_order[p.model]))
    record = {
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "worker_manifest": str(snapshot.resolve()),
        "worker_manifest_sha256": manifest_hash,
        "workers": args.jobs,
        "status": "running",
        "stop_on_failure": True,
        "archive_directory": str(args.archive_dir.resolve()),
        "archive_required_before_next_submission": True,
        "storage_preflight": storage,
        "operational_input_sha256": operational_hashes,
        "rows": {p.run_id: {"status": "pending"} for p in plans},
    }
    destination = args.out_dir / "queue_status.json"
    atomic_write_json(destination, record)

    def worker(plan):
        command = [
            sys.executable,
            str(root / "scripts/sweep.py"),
            "--manifest",
            str(snapshot.resolve()),
            "--run-id",
            plan.run_id,
            "--run",
        ]
        worker_log = args.out_dir / f"{plan.run_id}.log"
        with worker_log.open("x") as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        status_path = plan.output_root / plan.suite_id / plan.run_id / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        passed = result.returncode == 0 and status.get("status") == "completed"
        preservation = None
        if passed:
            completed_metadata(plan)
            try:
                if any(sha256(path) != digest for path, digest in operational_hashes.items()):
                    raise ValueError("Reviewed queue, archive helper, protocol or plan changed.")
                preservation = archive_completed(plan, snapshot, args.archive_dir,
                                                 worker_log, args.out_dir)
                if any(sha256(path) != digest for path, digest in operational_hashes.items()):
                    raise ValueError("Reviewed operational input changed during archiving.")
            except Exception as error:
                return {"status": "failed", "failure_stage": "artifact_preservation",
                        "managed_status": status, "training_completed": True,
                        "error": f"{type(error).__name__}: {error}",
                        "finished_utc": datetime.now(UTC).isoformat()}
        return {
            "status": "completed" if passed else "failed",
            "returncode": result.returncode,
            "finished_utc": datetime.now(UTC).isoformat(),
            "managed_status": status,
            "archive": preservation,
        }

    pending = iter(plans)
    stopped = False
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        active = {}

        def submit_next():
            nonlocal stopped
            if stopped:
                return
            try:
                unchanged = (
                    all(hashlib.sha256(path.read_bytes()).hexdigest() == manifest_hash
                        for path in (args.manifest, snapshot))
                    and all(sha256(path) == digest for path, digest in operational_hashes.items())
                )
            except OSError:
                unchanged = False
            if not unchanged:
                stopped = True
                record["stop_reason"] = (
                    "Manifest or reviewed operational input changed after preflight."
                )
                atomic_write_json(destination, record)
                return
            plan = next(pending, None)
            if plan is None:
                return
            record["rows"][plan.run_id] = {
                "status": "running",
                "started_utc": datetime.now(UTC).isoformat(),
            }
            active[pool.submit(worker, plan)] = plan
            atomic_write_json(destination, record)
            print(f"START {plan.run_id}", flush=True)

        for _ in range(args.jobs):
            submit_next()
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                plan = active.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                record["rows"][plan.run_id] = outcome
                stopped |= outcome["status"] != "completed"
                print(f"{outcome['status'].upper()} {plan.run_id}", flush=True)
            atomic_write_json(destination, record)
            if not stopped:
                for _ in finished:
                    submit_next()
    record["status"] = "stopped_after_failure" if stopped else "completed"
    record["finished_utc"] = datetime.now(UTC).isoformat()
    atomic_write_json(destination, record)
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
