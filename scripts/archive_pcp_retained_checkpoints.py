"""Copy a completed CPUv3 run's retained evidence into a verified, new archive."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from collections.abc import Mapping

import torch

SOURCE = "7124976af366d28d40a3ccf0ebce62aa2b4019783d13974b9f98dd1c316af262"
PROTOCOL_SHA = "48b22163ac8033f0e18c33e3d39dac907582d44f3864d9b03d5bc10694113f5a"
COMPARATOR_SHA = "35793515de2ed1ab450400ee0cdcd3eb7a9a675d18eb0cc3f701eb4352170c6c"
FRAMES = (120000, 240000, 360000, 480000, 600000)
GROUPS = {"adversary", "agent"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparator():
    path = Path(__file__).with_name("compare_pcp_retention_confirmation.py")
    need(sha(path) == COMPARATOR_SHA, "Reviewed retention comparator changed.")
    spec = importlib.util.spec_from_file_location("pcp_retention_archive_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(run):
    native = list(run.glob("benchmarl/*/checkpoints/checkpoint_*.pt"))
    need(
        len(native) == 5
        and {path.name for path in native} == {f"checkpoint_{frame}.pt" for frame in FRAMES},
        "All five unique 120k-through-600k native checkpoints are required.",
    )
    snapshots = list(run.glob("benchmarl/*/replay_diagnostics/*adversary_first_fallback*.pt"))
    need(len(snapshots) == 1, "Exactly one first predator fallback snapshot is required.")
    paths = [*native, *snapshots, run / "checkpoints/policy_state.pt"]
    paths.extend(
        run / name
        for name in ("metadata.json", "status.json", "resolved_config.yaml", "metrics.csv")
    )
    paths.extend(path for path in (run / "summary.json", run / "git.patch") if path.exists())
    for path in paths:
        need(
            path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run),
            f"Required artifact is missing, linked, or outside its run: {path}.",
        )
    return sorted(paths), snapshots[0]


def validate(run, paths, snapshot_path, reader):
    metadata = json.loads((run / "metadata.json").read_text())
    status = json.loads((run / "status.json").read_text())
    need(
        metadata.get("status") == status.get("status") == "completed"
        and type(metadata.get("frames")) is int
        and metadata["frames"] == 600000
        and type(metadata.get("iterations")) is int
        and metadata["iterations"] == 100,
        "Archive requires a completed 600k/100-iteration managed run.",
    )
    need(
        metadata.get("source_sha256") == SOURCE
        and metadata.get("model") == "pcp_comm_identity"
        and metadata.get("seed") in (10, 11)
        and type(metadata.get("seed")) is int
        and metadata.get("protocol", {}).get("protocol_id") == "pcp_corrected_cpu_v3"
        and metadata["protocol"].get("protocol_sha256") == PROTOCOL_SHA
        and metadata["protocol"].get("source_sha256") == SOURCE
        and metadata["protocol"].get("stage") == "confirmation",
        "Archive scope must match the frozen CPUv3 Identity confirmation.",
    )
    native_headers = []
    for frame in FRAMES:
        path, payload = reader.native_checkpoint(run, frame)
        native_headers.append(
            {
                "relative_path": str(path.relative_to(run)),
                "frames": payload["state"]["total_frames"],
                "iterations": payload["state"]["n_iters_performed"],
            }
        )
    final = torch.load(run / "checkpoints/policy_state.pt", map_location="cpu", weights_only=False)
    need(
        final.get("schema_version") == 1 and final.get("frames") == 600000,
        "Final policy checkpoint is not the completed 600k policy.",
    )
    for field in ("run_id", "suite_id", "algorithm", "task", "model", "seed"):
        need(
            field in metadata and final.get(field) == metadata[field],
            f"Final policy checkpoint identity differs: {field}.",
        )
    need(
        isinstance(final.get("group_policies"), Mapping) and set(final["group_policies"]) == GROUPS,
        "Final policy checkpoint must contain both policy groups.",
    )
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    need(
        snapshot.get("schema_version") == 1
        and snapshot.get("diagnostic_only") is True
        and snapshot.get("not_resume") is True
        and snapshot.get("reason") == "first_fallback"
        and snapshot.get("frames") == 564000
        and snapshot.get("group") == "adversary"
        and snapshot.get("source_sha256") == SOURCE
        and isinstance(snapshot.get("group_policies"), Mapping)
        and set(snapshot["group_policies"]) == GROUPS,
        "Snapshot must be the preserved diagnostic-only first predator fallback at 564k.",
    )
    return metadata, native_headers


def copy_exclusive(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def write_exclusive(path, document):
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def archive_run(run, destination):
    run, destination = run.resolve(), destination.resolve()
    need(
        not destination.is_relative_to(run), "Archive destination must be outside the original run."
    )
    need(not destination.exists(), "Archive destination already exists; overwrites are forbidden.")
    reader = comparator()
    paths, snapshot_path = inventory(run)
    before = {path: sha(path) for path in paths}
    metadata, native_headers = validate(run, paths, snapshot_path, reader)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive reservation: never reuse or overwrite an archive.
    try:
        for source in paths:
            copy_exclusive(source, destination / source.relative_to(run))
        entries = []
        for source in paths:
            target = destination / source.relative_to(run)
            source_after, copied = sha(source), sha(target)
            need(
                before[source] == source_after == copied,
                f"Source changed or archive readback differs: {source}.",
            )
            entries.append(
                {
                    "source_path": str(source),
                    "relative_path": str(source.relative_to(run)),
                    "archive_path": str(target),
                    "bytes": source.stat().st_size,
                    "source_sha256_before": before[source],
                    "source_sha256_after": source_after,
                    "archive_sha256": copied,
                }
            )
        manifest = {
            "schema_version": 1,
            "archive": "pcp_retained_checkpoints_v3",
            "status": "verified",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "source_run": str(run),
            "archive_directory": str(destination),
            "run_id": metadata["run_id"],
            "seed": metadata["seed"],
            "frames": metadata["frames"],
            "source_sha256": SOURCE,
            "scientific_config_sha256": metadata.get("scientific_config_sha256"),
            "protocol": metadata["protocol"],
            "native_checkpoint_headers": native_headers,
            "snapshot_frames": 564000,
            "snapshot_group": "adversary",
            "files": entries,
            "file_count": len(entries),
            "script_sha256": sha(Path(__file__)),
            "reader_sha256": COMPARATOR_SHA,
            "approval_decision": "not_issued",
            "limits": "Verified evidence copy only; full invariance, replay and scientific reviews "
            "remain separate. Diagnostic snapshots and native checkpoints are not exact "
            "resume points.",
        }
        pending = destination / "archive_manifest.pending.json"
        write_exclusive(pending, manifest)
        # A successful fsync precedes publication. link() atomically creates
        # the final name and refuses to replace any existing file.
        os.link(pending, destination / "archive_manifest.json")
        try:
            pending.unlink()
        except OSError:
            pass  # The verified final manifest is already safely published.
        return manifest
    except Exception as error:
        # Preserve a visibly failed partial copy, never a misleading completion
        # manifest. A later attempt must use a new destination.
        try:
            write_exclusive(
                destination / "archive_failure.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "source_run": str(run),
                    "error": f"{type(error).__name__}: {error}",
                    "source_sha256_before": {str(path): digest for path, digest in before.items()},
                },
            )
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = archive_run(args.run_dir, args.destination)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        return 2
    path = Path(manifest["archive_directory"]) / "archive_manifest.json"
    print(json.dumps({"status": "verified", "manifest": str(path), "manifest_sha256": sha(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
