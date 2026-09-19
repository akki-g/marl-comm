"""Verified no-overwrite copies of completed PCP retained checkpoint evidence."""

import importlib.util
import json

import pytest
import torch


@pytest.fixture
def archive_module(config_root):
    path = config_root.parent / "scripts/archive_pcp_retained_checkpoints.py"
    spec = importlib.util.spec_from_file_location("pcp_checkpoint_archiver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed_run(archive_module, tmp_path):
    run = tmp_path / "original_run"
    run.mkdir()
    identity = {
        "run_id": run.name,
        "suite_id": "pcp_candidate_confirmation_cpu_v3",
        "algorithm": "mappo",
        "task": "vmas_predator_capture_prey",
        "model": "pcp_comm_identity",
        "seed": 10,
    }
    metadata = {
        **identity,
        "status": "completed",
        "frames": 600000,
        "iterations": 100,
        "source_sha256": archive_module.SOURCE,
        "scientific_config_sha256": "a" * 64,
        "protocol": {
            "protocol_id": "pcp_corrected_cpu_v3",
            "protocol_sha256": archive_module.PROTOCOL_SHA,
            "source_sha256": archive_module.SOURCE,
            "stage": "confirmation",
        },
    }
    (run / "metadata.json").write_text(json.dumps(metadata))
    (run / "status.json").write_text(json.dumps({"status": "completed"}))
    (run / "resolved_config.yaml").write_text("scientific_configuration: preserved verbatim\n")
    (run / "metrics.csv").write_text("metric,value\nreturn,1\n")
    groups = {name: {"signed_zero": torch.tensor([-0.0, 0.0])} for name in ("adversary", "agent")}
    folder = run / "benchmarl/native/checkpoints"
    folder.mkdir(parents=True)
    for frame in archive_module.FRAMES:
        torch.save(
            {
                "state": {"total_frames": frame, "n_iters_performed": frame // 6000},
                "loss_adversary": groups["adversary"],
                "loss_agent": groups["agent"],
            },
            folder / f"checkpoint_{frame}.pt",
        )
    (run / "checkpoints").mkdir()
    torch.save(
        {**identity, "schema_version": 1, "frames": 600000, "group_policies": groups},
        run / "checkpoints/policy_state.pt",
    )
    snapshots = run / "benchmarl/native/replay_diagnostics"
    snapshots.mkdir()
    torch.save(
        {
            "schema_version": 1,
            "diagnostic_only": True,
            "not_resume": True,
            "reason": "first_fallback",
            "frames": 564000,
            "group": "adversary",
            "source_sha256": archive_module.SOURCE,
            "group_policies": groups,
        },
        snapshots / "frame000564000_adversary_first_fallback.pt",
    )
    return run


def test_archive_preserves_exact_bytes_including_signed_zero_and_original_hashes(
    archive_module, completed_run, tmp_path
):
    destination = tmp_path / "archive"
    manifest = archive_module.archive_run(completed_run, destination)
    assert manifest["status"] == "verified" and manifest["file_count"] == 11
    assert manifest["approval_decision"] == "not_issued"
    assert len(manifest["native_checkpoint_headers"]) == 5
    for item in manifest["files"]:
        assert item["source_sha256_before"] == item["source_sha256_after"] == item["archive_sha256"]
        assert (destination / item["relative_path"]).read_bytes() == (
            completed_run / item["relative_path"]
        ).read_bytes()
    payload = torch.load(
        destination / "benchmarl/native/checkpoints/checkpoint_480000.pt", weights_only=False
    )
    assert torch.signbit(payload["loss_agent"]["signed_zero"]).tolist() == [True, False]
    assert json.loads((destination / "archive_manifest.json").read_text()) == manifest


@pytest.mark.parametrize(
    "corruption",
    ["missing480k", "native_header", "final_seed", "snapshot_frame", "running", "frames"],
)
def test_archive_rejects_missing_or_misidentified_evidence_before_creating_destination(
    archive_module, completed_run, tmp_path, corruption
):
    if corruption in ("missing480k", "native_header"):
        path = completed_run / "benchmarl/native/checkpoints/checkpoint_480000.pt"
        if corruption == "missing480k":
            path.unlink()
        else:
            payload = torch.load(path, weights_only=False)
            payload["state"]["n_iters_performed"] = 79
            torch.save(payload, path)
    elif corruption in ("final_seed", "snapshot_frame"):
        path = completed_run / (
            "checkpoints/policy_state.pt"
            if corruption == "final_seed"
            else "benchmarl/native/replay_diagnostics/frame000564000_adversary_first_fallback.pt"
        )
        payload = torch.load(path, weights_only=False)
        payload["seed" if corruption == "final_seed" else "frames"] = (
            11 if corruption == "final_seed" else 558000
        )
        torch.save(payload, path)
    else:
        path = completed_run / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["status" if corruption == "running" else "frames"] = (
            "running" if corruption == "running" else 594000
        )
        path.write_text(json.dumps(metadata))
    destination = tmp_path / "archive"
    with pytest.raises(ValueError):
        archive_module.archive_run(completed_run, destination)
    assert not destination.exists()


def test_archive_never_overwrites_or_merges_with_existing_destination(
    archive_module, completed_run, tmp_path
):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "important.pt"
    marker.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="overwrites"):
        archive_module.archive_run(completed_run, destination)
    assert marker.read_bytes() == b"preserve"
    assert list(destination.iterdir()) == [marker]


@pytest.mark.parametrize("corruption", ["copy_bytes", "copy_signed_zero", "source_changed"])
def test_archive_rejects_corrupt_copy_or_source_mutation_and_marks_partial_copy_failed(
    archive_module, completed_run, tmp_path, monkeypatch, corruption
):
    original_copy = archive_module.copy_exclusive
    triggered = False

    def corrupt(source, target):
        nonlocal triggered
        original_copy(source, target)
        if triggered or source.name != "checkpoint_480000.pt":
            return
        triggered = True
        if corruption == "copy_bytes":
            with target.open("ab") as stream:
                stream.write(b"corruption")
        elif corruption == "copy_signed_zero":
            payload = torch.load(target, weights_only=False)
            payload["loss_agent"]["signed_zero"][0] = 0.0
            torch.save(payload, target)
        else:
            with source.open("ab") as stream:
                stream.write(b"source changed while archiving")

    monkeypatch.setattr(archive_module, "copy_exclusive", corrupt)
    destination = tmp_path / "partial"
    with pytest.raises(ValueError, match="Source changed or archive readback differs"):
        archive_module.archive_run(completed_run, destination)
    assert not (destination / "archive_manifest.json").exists()
    assert json.loads((destination / "archive_failure.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="overwrites"):
        archive_module.archive_run(completed_run, destination)


def test_manifest_fsync_failure_never_publishes_verified_completion(
    archive_module, completed_run, tmp_path, monkeypatch
):
    original = archive_module.write_exclusive

    def fail_after_manifest_write(path, document):
        original(path, document)
        if path.name == "archive_manifest.pending.json":
            raise OSError("injected manifest fsync failure after complete JSON write")

    monkeypatch.setattr(archive_module, "write_exclusive", fail_after_manifest_write)
    destination = tmp_path / "partial_manifest"
    with pytest.raises(OSError, match="fsync failure"):
        archive_module.archive_run(completed_run, destination)
    assert not (destination / "archive_manifest.json").exists()
    assert json.loads((destination / "archive_failure.json").read_text())["status"] == "failed"
