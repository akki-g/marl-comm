"""A failed scientific row must stop spending the remaining seed allocation."""

from __future__ import annotations

import importlib.util
import json
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def sufficient_mock_storage(monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=12 * 1024**3))


def queue_module():
    path = Path(__file__).resolve().parents[2] / "scripts/run_pcp_budget_queue.py"
    spec = importlib.util.spec_from_file_location("pcp_budget_queue_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plans(tmp_path):
    return [SimpleNamespace(
        seed=seed, ablation_value=visibility, model=model, max_n_frames=600000,
        ablation="visibility", protocol={"stage": "comparison"},
        output_root=tmp_path / "runs", suite_id="study", run_id=f"{seed}-{visibility}-{model}",
    ) for seed in range(20, 25) for visibility in ("radius1", "global")
        for model in ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0",
                      "pcp_broadcast_k3")]


def test_first_failure_preserves_thirty_nine_unspent_rows(tmp_path, monkeypatch):
    module = queue_module()
    rows = plans(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("test manifest")
    monkeypatch.setattr(module, "read_manifest", lambda path: rows)
    monkeypatch.setattr(module, "validate_plan", lambda *a, **kw: None)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(module.subprocess, "run", fail)
    output = tmp_path / "queue"
    assert module.main([
        "--manifest", str(manifest), "--out-dir", str(output),
        "--archive-dir", str(tmp_path / "archives"), "--jobs", "1",
    ]) == 2
    assert len(calls) == 1
    report = json.loads((output / "queue_status.json").read_text())
    assert report["status"] == "stopped_after_failure"
    states = [row["status"] for row in report["rows"].values()]
    assert states.count("failed") == 1 and states.count("pending") == 39
    assert len(list(output.glob("*.log"))) == 1


@pytest.mark.parametrize("defect", ["missing_seed", "duplicate", "shortened", "existing_attempt",
                                  "run_id_collision", "split_suite", "existing_archive"])
def test_invalid_or_existing_allocation_never_starts_worker(tmp_path, monkeypatch, defect):
    module = queue_module()
    rows = plans(tmp_path)
    if defect == "missing_seed":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = rows[0]
    elif defect == "shortened":
        rows[0].max_n_frames = 6000
    elif defect == "existing_attempt":
        (rows[0].output_root / rows[0].suite_id / rows[0].run_id).mkdir(parents=True)
    elif defect == "run_id_collision":
        rows[-1].run_id = rows[0].run_id
    elif defect == "split_suite":
        rows[-1].suite_id = "another_study"
    else:
        (tmp_path / "archives" / rows[0].run_id).mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("test manifest")
    monkeypatch.setattr(module, "read_manifest", lambda path: rows)
    monkeypatch.setattr(module, "validate_plan", lambda *a, **kw: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("started worker"))
    with pytest.raises(SystemExit):
        module.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "queue"),
                     "--archive-dir", str(tmp_path / "archives")])
    assert not (tmp_path / "queue").exists()


@pytest.mark.parametrize("defect", ["short_completion", "wrong_identity", "manifest_change"])
def test_terminal_false_positive_or_manifest_change_stops_remaining_budget(
    tmp_path, monkeypatch, defect,
):
    module = queue_module()
    rows = plans(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("immutable test manifest")
    monkeypatch.setattr(module, "read_manifest", lambda path: rows)
    monkeypatch.setattr(module, "validate_plan", lambda *a, **kw: None)
    calls = []

    def complete(command, **kwargs):
        calls.append(command)
        plan = next(p for p in rows if p.run_id == command[-2])
        run = plan.output_root / plan.suite_id / plan.run_id
        run.mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({"status": "completed"}))
        metadata = {
            "status": "completed", "frames": 600000, "iterations": 100,
            "run_id": plan.run_id, "suite_id": plan.suite_id,
            "task": "vmas_predator_capture_prey", "algorithm": "mappo",
            "model": plan.model, "seed": plan.seed, "ablation": "visibility",
            "ablation_value": plan.ablation_value, "execution_purpose": "scientific",
        }
        if defect == "short_completion":
            metadata.update(frames=594000, iterations=99)
        elif defect == "wrong_identity":
            metadata["model"] = "wrong_model"
        else:
            manifest.write_text("changed during first worker")
        (run / "metadata.json").write_text(json.dumps(metadata))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", complete)
    monkeypatch.setattr(module, "archive_completed", lambda *a, **kw: {"checks_passed": True})
    output = tmp_path / "queue"
    assert module.main([
        "--manifest", str(manifest), "--out-dir", str(output),
        "--archive-dir", str(tmp_path / "archives"), "--jobs", "1",
    ]) == 2
    assert len(calls) == 1
    report = json.loads((output / "queue_status.json").read_text())
    assert report["status"] == "stopped_after_failure"
    assert sum(row["status"] == "pending" for row in report["rows"].values()) == 39
    assert Path(calls[0][3]).read_text() == "immutable test manifest"


def record_completed(plan):
    run = plan.output_root / plan.suite_id / plan.run_id
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": "completed"}))
    (run / "metadata.json").write_text(json.dumps({
        "status": "completed", "frames": 600000, "iterations": 100,
        "run_id": plan.run_id, "suite_id": plan.suite_id,
        "task": "vmas_predator_capture_prey", "algorithm": "mappo",
        "model": plan.model, "seed": plan.seed, "ablation": "visibility",
        "ablation_value": plan.ablation_value, "execution_purpose": "scientific",
    }))


@pytest.mark.parametrize("archive_result", ["verified", "exit_failure", "missing_receipt",
                                          "wrong_run", "corruption", "false_checks",
                                          "changed_sources", "failure_marker", "helper_changed"])
def test_queue_requires_verified_archive_before_spending_next_row(
    tmp_path, monkeypatch, archive_result,
):
    module = queue_module()
    rows = plans(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("fixed manifest")
    monkeypatch.setattr(module, "read_manifest", lambda path: rows)
    monkeypatch.setattr(module, "validate_plan", lambda *a, **kw: None)
    events, worker_streams = [], []
    original_hash = module.sha256

    def hash_with_drift(path):
        if (archive_result == "helper_changed" and events
                and Path(path).name == "archive_pcp_budget_run.py"):
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(module, "sha256", hash_with_drift)

    def run(command, **kwargs):
        rid = command[command.index("--run-id") + 1]
        plan = next(p for p in rows if p.run_id == rid)
        if Path(command[1]).name == "sweep.py":
            events.append("training")
            if len(events) > 1:
                return SimpleNamespace(returncode=1)
            worker_streams.append(kwargs["stdout"])
            kwargs["stdout"].write(f"[sweep] {rid} -> completed\n")
            record_completed(plan)
            return SimpleNamespace(returncode=0)
        events.append("archive")
        assert worker_streams[0].closed
        worker = Path(command[command.index("--worker-log") + 1])
        assert worker.read_text() == f"[sweep] {rid} -> completed\n"
        if archive_result == "exit_failure":
            return SimpleNamespace(returncode=1)
        destination = Path(command[command.index("--destination") + 1])
        destination.mkdir(parents=True)
        archive = destination / "run.tar.gz"
        archive.write_bytes(b"verified gzip placeholder for orchestration test")
        receipt = {"schema_version": 1, "checks_passed": True, "run_id": rid,
                   "archive_file": "run.tar.gz", "source_before_after_equal": True,
                   "all_archive_files_rehashed": True, "original_files_removed": False,
                   "archive_bytes": archive.stat().st_size,
                   "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        if archive_result == "wrong_run":
            receipt["run_id"] = "another_run"
        elif archive_result == "corruption":
            archive.write_bytes(b"corrupt")
        elif archive_result == "false_checks":
            receipt["checks_passed"] = False
        elif archive_result == "changed_sources":
            receipt["source_before_after_equal"] = False
        elif archive_result == "failure_marker":
            (destination / "archive_failure.json").write_text("{}")
        if archive_result != "missing_receipt":
            (destination / "archive_manifest.json").write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "queue"
    assert module.main(["--manifest", str(manifest), "--out-dir", str(output),
                        "--archive-dir", str(tmp_path / "archives"), "--jobs", "1"]) == 2
    report = json.loads((output / "queue_status.json").read_text())
    first = report["rows"][rows[0].run_id]
    assert report["storage_preflight"]["checks_passed"]
    assert report["archive_required_before_next_submission"]
    if archive_result == "verified":
        assert events == ["training", "archive", "training"]
        assert first["status"] == "completed" and first["archive"]["checks_passed"]
        assert sum(r["status"] == "pending" for r in report["rows"].values()) == 38
    else:
        assert events == (["training"] if archive_result == "helper_changed"
                          else ["training", "archive"])
        assert first["status"] == "failed" and first["training_completed"]
        assert first["failure_stage"] == "artifact_preservation"
        assert first["managed_status"]["status"] == "completed"
        assert sum(r["status"] == "pending" for r in report["rows"].values()) == 39


@pytest.mark.parametrize("free", [8 * 1024**3 - 1, 8 * 1024**3])
def test_storage_preflight_requires_eight_gib_before_any_training(tmp_path, monkeypatch, free):
    module = queue_module()
    rows = plans(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("fixed manifest")
    monkeypatch.setattr(module, "read_manifest", lambda path: rows)
    monkeypatch.setattr(module, "validate_plan", lambda *a, **kw: None)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw:
                        (calls.append(a), SimpleNamespace(returncode=1))[1])
    output = tmp_path / "queue"
    args = ["--manifest", str(manifest), "--out-dir", str(output),
            "--archive-dir", str(tmp_path / "archives"), "--jobs", "1"]
    if free < module.MINIMUM_FREE_BYTES:
        with pytest.raises(SystemExit):
            module.main(args)
        assert calls == [] and not output.exists()
    else:
        assert module.main(args) == 2 and len(calls) == 1
        report = json.loads((output / "queue_status.json").read_text())
        assert all(p["free_bytes"] == p["required_free_bytes"] == free
                   for p in report["storage_preflight"]["paths"])
