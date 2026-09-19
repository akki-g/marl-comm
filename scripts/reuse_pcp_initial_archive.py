"""Preserve the two original D4 archives without changing historical approval bindings.

Only the original seed-20/radius1 Identity and LocalCapacity archives are eligible.
All original bytes and paths remain inside their original archive; a separate receipt
binds byte-identical copies to the verified operational relocation manifest.
"""

from __future__ import annotations

from datetime import datetime, UTC
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


OLD_ROOT = Path("/Users/akshatguduru/Desktop/Thesis/marl-comm")
SUITE = "pcp_visibility_budget_cpu_v1"
SOURCE = "69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e"
PROTOCOL = "ec7c109413625f270b79bffc1753adf5b7691d008eb90a4297d844b020898b0a"
ANCHORS = {
    "pcp_comm_identity": {
        "archive": "9e0120d070db3a37723fc440a8e9ef5dbd90c7bb23288147ff603ba3c10593b1",
        "receipt": "43681456fe9096e2faa457369b389eb25d549368dbe6e925bbca38c716647cc7",
    },
    "pcp_local_capacity": {
        "archive": "fbed25c0522a9b67fcd4f701b20603bbcac561206d5eed7a82701ed5167e9a8e",
        "receipt": "6a0e1934f1a90782e90ef3e41d210d5d8d115e29796cb01b4dbf3949320ade5a",
    },
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ordinary(path, *, directory=False, absent=False):
    path = Path(path)
    need(path.is_absolute() and ".." not in path.parts, "Use ordinary absolute paths.")
    need(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlink path: {path}")
    if not absent:
        need(path.is_dir() if directory else path.is_file(), f"Missing ordinary input: {path}")
    return path


def verify_archive(path):
    """Stream every member and gzip trailer, rejecting duplicates and nonregular files."""
    actual, document, embedded_sha = {}, None, None
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = member.name
            need(member.isfile() and not PurePosixPath(name).is_absolute()
                 and ".." not in PurePosixPath(name).parts and name not in actual,
                 f"Unsafe or duplicate archive member: {name}")
            stream = archive.extractfile(member)
            need(stream is not None, "Unreadable archive member.")
            if name == "ARCHIVE_MANIFEST.json":
                payload = stream.read()
                document = json.loads(payload)
                member_sha = embedded_sha = hashlib.sha256(payload).hexdigest()
            else:
                member_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            actual[name] = {"sha256": member_sha, "bytes": member.size, "mode": member.mode}
    need(isinstance(document, dict) and document.get("schema_version") == 1,
         "Missing original archive inventory.")
    need(set(actual) == set(document["files"]) | {"ARCHIVE_MANIFEST.json"},
         "Archive member inventory differs.")
    for name, entry in document["files"].items():
        need(all(actual[name][key] == entry[key] for key in ("sha256", "bytes", "mode")),
             f"Archived bytes differ from inventory: {name}")
    with gzip.open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    return document, embedded_sha


def verify_original(root, plan, old_queue):
    root = ordinary(root, directory=True)
    old_queue = ordinary(old_queue, directory=True)
    expected_id = (f"predator_capture_prey__mappo__{plan.model}__{SUITE}"
                   "__visibility__radius1__seed020")
    need(plan.model in ANCHORS and plan.run_id == expected_id and plan.seed == 20
         and plan.ablation_value == "radius1" and plan.suite_id == SUITE
         and plan.max_n_frames == 600000 and plan.attempt == 0 and plan.retry_of is None,
         "Archive reuse is limited to the two original completed controls.")
    run = ordinary(plan.output_root / plan.suite_id / plan.run_id, directory=True)
    need(run.is_relative_to(root), "Reused run lies outside relocated repository.")
    source = ordinary(root / "results" / SUITE / "interim_hold_archives" / plan.run_id,
                      directory=True)
    archive, receipt_path = source / "run.tar.gz", source / "archive_manifest.json"
    need({p.name for p in source.iterdir()} == {archive.name, receipt_path.name},
         "Original archive directory contains unexpected evidence.")
    ordinary(archive)
    ordinary(receipt_path)
    anchor = ANCHORS[plan.model]
    need(sha(archive) == anchor["archive"] and sha(receipt_path) == anchor["receipt"],
         "Original archive or receipt differs from reviewed immutable bytes.")
    document, embedded_sha = verify_archive(archive)
    identity = {"run_id": plan.run_id, "suite_id": SUITE, "source_sha256": SOURCE,
                "protocol_sha256": PROTOCOL, "model": plan.model, "seed": 20,
                "visibility": "radius1", "frames": 600000, "iterations": 100}
    need(all(document.get(k) == v for k, v in identity.items()),
         "Original archive identity differs.")
    receipt = json.loads(receipt_path.read_text())
    expected = {"schema_version": 1, "checks_passed": True, "run_id": plan.run_id,
                "archive_file": "run.tar.gz", "archive_sha256": anchor["archive"],
                "archive_bytes": archive.stat().st_size, "source_before_after_equal": True,
                "all_archive_files_rehashed": True, "original_files_removed": False,
                "embedded_manifest_sha256": embedded_sha,
                "archived_files": len(document["files"]),
                "run_file_count": document["run_file_count"],
                "uncompressed_file_bytes": sum(v["bytes"] for v in document["files"].values())}
    need(receipt == expected, "Original archive receipt does not match independent read-back.")
    inputs, run_inputs, mapped = {}, {}, {}
    for name, entry in document["files"].items():
        old = Path(entry["source_path"])
        need(old.is_relative_to(OLD_ROOT) and ".." not in old.parts,
             f"Unexpected historical source path: {old}")
        path = ordinary(root / old.relative_to(OLD_ROOT))
        need(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
             f"Historical archive input changed: {path}")
        inputs[str(path)] = entry["sha256"]
        mapped[name] = path
        if name.startswith("run/"):
            need(path == run / name.removeprefix("run/"), "Archived run path differs.")
            run_inputs[str(path)] = entry["sha256"]
    actual = set()
    for path in run.rglob("*"):
        ordinary(path, directory=path.is_dir())
        if path.is_file():
            actual.add(str(path))
    need(actual == set(run_inputs) and len(actual) == document["run_file_count"],
         "Current raw run inventory differs from original archive.")
    metadata = json.loads((run / "metadata.json").read_text())
    approval = json.loads(mapped["provenance/launch_approval.json"].read_text())
    need(metadata["protocol"]["approval_sha256"] == content_sha(approval),
         "Original metadata is not bound to its preserved approval.")
    need(metadata["scientific_config_sha256"] == plan.scientific_hash,
         "Reused archive scientific configuration differs from the relocated row.")
    log = ordinary(old_queue / f"{plan.run_id}.log")
    need(sha(log) == document["files"]["worker/stdout.log"]["sha256"],
         "Original queue log differs from archived closure log.")
    need(f"[sweep] {plan.run_id} -> completed" in log.read_text(),
         "Original queue log lacks successful closure.")
    inputs.update({str(archive): anchor["archive"], str(receipt_path): anchor["receipt"],
                   str(log): sha(log)})
    need(all(sha(Path(path)) == digest for path, digest in inputs.items()),
         "An original input changed during archive verification.")
    return {"checks_passed": True, "run_id": plan.run_id, "source": str(source),
            "archive_sha256": anchor["archive"], "receipt_sha256": anchor["receipt"],
            "embedded_manifest_sha256": embedded_sha, "archive_bytes": archive.stat().st_size,
            "archived_files": len(document["files"]), "run_file_count": len(run_inputs),
            "all_archive_files_rehashed": True, "source_before_after_equal": True,
            "original_approval_matches_metadata": True, "input_artifact_sha256": inputs}


def copy_verified(root, plan, old_queue, manifest, relocation_receipt, destination, expected):
    """Publish two unchanged historical files and a separate operational reuse receipt."""
    destination = ordinary(destination, absent=True)
    need(not destination.exists(), "Preserve existing archive destination; no retry or overwrite.")
    manifest, relocation_receipt = ordinary(manifest), ordinary(relocation_receipt)
    run = plan.output_root / plan.suite_id / plan.run_id
    source = Path(expected["source"])
    for protected in (run, source, old_queue, manifest.parent, relocation_receipt.parent):
        need(not destination.is_relative_to(protected)
             and not protected.is_relative_to(destination),
             "Archive reuse destination overlaps protected inputs.")
    binding = {str(manifest): sha(manifest), str(relocation_receipt): sha(relocation_receipt)}
    relocation = json.loads(relocation_receipt.read_text())
    need(relocation["generated_file_sha256"][manifest.name] == binding[str(manifest)]
         and relocation["source_sha256"] == SOURCE and relocation["checks_passed"] is True,
         "Derived manifest does not match the verified relocation receipt.")
    need(verify_original(root, plan, old_queue) == expected,
         "Original archive inputs differ from preflight.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    for name, digest in (("run.tar.gz", expected["archive_sha256"]),
                         ("archive_manifest.json", expected["receipt_sha256"])):
        with (source / name).open("rb") as original, (destination / name).open("xb") as copied:
            shutil.copyfileobj(original, copied, length=1024 * 1024)
            copied.flush()
            os.fsync(copied.fileno())
        need(sha(destination / name) == digest, f"Archive copy differs: {name}")
    document, embedded_sha = verify_archive(destination / "run.tar.gz")
    need(embedded_sha == expected["embedded_manifest_sha256"]
         and len(document["files"]) == expected["archived_files"],
         "Copied archive read-back differs from preserved inventory.")
    need(verify_original(root, plan, old_queue) == expected
         and all(sha(Path(path)) == digest for path, digest in binding.items()),
         "Original or relocation inputs changed during archive copy.")
    reuse = {"schema_version": 1, "recorded_utc": datetime.now(UTC).isoformat(),
             "checks_passed": True, "run_id": plan.run_id, "destination": str(destination),
             "source_verification": expected, "operational_binding_sha256": binding,
             "original_files_removed": False, "historical_files_modified": False,
             "all_copied_files_rehashed": True, "copied_archive_readback_verified": True,
             "scope": "Byte-identical original completed training archive. Current held-out banks "
                      "and their strict reviews remain separate coordinator evidence.",
             "reason": "Historical approval hash remains in original metadata/archive; the "
                       "relocated operational manifest is bound only by this separate receipt."}
    path = destination / "archive_reuse_receipt.json"
    with path.open("x") as stream:
        json.dump(reuse, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"destination": str(destination), "receipt_sha256": expected["receipt_sha256"],
            "archive_sha256": expected["archive_sha256"],
            "archive_bytes": expected["archive_bytes"],
            "checks_passed": True, "reused_original_archive": True,
            "reuse_receipt_sha256": sha(path)}
