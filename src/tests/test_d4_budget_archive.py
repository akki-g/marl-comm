"""Synthetic preservation tests; these fixtures never represent study evidence."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import io
import itertools
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from commstudy.experiments.protocols import content_sha256, load_protocol
from commstudy.experiments.sweeps import RunPlan, write_manifest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "budget_archive", ROOT / "scripts/archive_pcp_budget_run.py"
)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "SYNTHETIC-repository"
    protocol = load_protocol(ROOT / "configs/protocols/pcp_visibility_budget_cpu_v1.yaml")
    protocol["status"] = "frozen"
    assert content_sha256(protocol) == archive.PROTOCOL_SHA
    monkeypatch.setattr(archive, "ROOT", root)
    monkeypatch.setattr(archive, "source_fingerprint", lambda _root: archive.SOURCE)
    monkeypatch.setattr(archive, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(archive, "validate_plan", lambda *args, **kwargs: None)
    plans = []
    for model, visibility, seed in itertools.product(
        archive.MODELS, ("radius1", "global"), range(20, 25)
    ):
        plans.append(
            RunPlan(
                run_id=f"SYNTHETIC-{model}-{visibility}-{seed}",
                suite_id="SYNTHETIC-comparison",
                task="vmas_predator_capture_prey",
                algorithm="mappo",
                model=model,
                seed=seed,
                ablation="visibility",
                ablation_value=visibility,
                max_n_frames=600000,
                status="completed",
                output_root=root / "runs",
                overrides=(),
                command="SYNTHETIC",
                scientific_hash="b" * 64,
                model_contract={"task_contract": {}, "parameter_counts": {}},
                protocol={
                    "source_sha256": archive.SOURCE,
                    "sha256": archive.PROTOCOL_SHA,
                    "stage": "comparison",
                    "path": "configs/protocol.yaml",
                    "approval": "configs/approval.json",
                },
            )
        )
    manifest = root / "queue/manifest.csv"
    manifest.parent.mkdir(parents=True)
    write_manifest(manifest, plans)
    plan = plans[0]
    run = plan.output_root / plan.suite_id / plan.run_id
    approval = {"fixture": "SYNTHETIC launch approval dependency"}
    dump(root / "configs/approval.json", approval)
    dump(root / "configs/protocol.yaml", {"fixture": "SYNTHETIC serialized protocol"})
    metadata = {
        "run_id": plan.run_id,
        "suite_id": plan.suite_id,
        "task": plan.task,
        "algorithm": "mappo",
        "model": plan.model,
        "seed": plan.seed,
        "ablation": "visibility",
        "ablation_value": "radius1",
        "status": "completed",
        "frames": 600000,
        "iterations": 100,
        "execution_purpose": "scientific",
        "source_sha256": archive.SOURCE,
        "scientific_config_sha256": "b" * 64,
        "task_runtime_contract": {},
        "parameters": {},
        "versions": {"python": protocol["runtime"]["python"], **protocol["runtime"]["packages"]},
        "runtime": {
            "train_device": "cpu",
            "sampling_device": "cpu",
            "buffer_device": "cpu",
            "torch_num_threads": 1,
        },
        "protocol": {
            "protocol_id": archive.PROTOCOL_ID,
            "protocol_sha256": archive.PROTOCOL_SHA,
            "source_sha256": archive.SOURCE,
            "stage": "comparison",
            "factors": {"model": plan.model, "visibility": "radius1"},
            "approval_sha256": content_sha256(approval),
        },
    }
    dump(run / "metadata.json", metadata)
    dump(run / "status.json", {"status": "completed"})
    dump(run / "summary.json", {"fixture": "SYNTHETIC summary"})
    (run / "metrics.csv").write_bytes(b"SYNTHETIC metrics\n" * 10000)
    (run / "resolved_config.yaml").write_text("# SYNTHETIC settings\n")
    (run / "empty.dat").write_bytes(b"")
    (run / "arbitrary-binary.dat").write_bytes(bytes(range(256)) * 20)
    checkpoint = {
        "schema_version": 1,
        **{
            key: metadata[key]
            for key in ("run_id", "suite_id", "task", "algorithm", "model", "seed", "frames")
        },
    }
    (run / "checkpoints").mkdir()
    torch.save(checkpoint, run / "checkpoints/policy_state.pt")
    native = {
        "checkpoints": [],
        "replay_diagnostics": [],
        "final_native_actor_matches_managed": True,
        "managed_final_policy_sha256": archive.sha256(run / "checkpoints/policy_state.pt"),
    }
    for frame in range(120000, 600001, 120000):
        path = run / f"benchmarl/SYNTHETIC/checkpoints/checkpoint_{frame}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC native dependency {frame}".encode())
        native["checkpoints"].append(
            {
                "path": str(path.resolve()),
                "sha256": archive.sha256(path),
                "frames": frame,
                "iterations": frame // 6000,
            }
        )
    diagnostic = run / "benchmarl/SYNTHETIC/replay_diagnostics/snapshot.pt"
    diagnostic.parent.mkdir()
    diagnostic.write_bytes(b"SYNTHETIC diagnostic, never a resume point")
    native["replay_diagnostics"].append(
        {"path": str(diagnostic.resolve()), "sha256": archive.sha256(diagnostic)}
    )
    # Native-header/actor validation is independently tested through the actual helper.
    monkeypatch.setattr(archive, "inventory_reader", lambda: lambda _run: native)
    for path in (
        "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
        "scripts/evaluate_pcp_budget.py",
        "results/pcp_visibility_budget_cpu_v1/package_source_freeze.json",
        "src/commstudy/SYNTHETIC.py",
        "pyproject.toml",
        "uv.lock",
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("SYNTHETIC archival dependency\n")
    log = manifest.parent / f"{plan.run_id}.log"
    log.write_text(f"SYNTHETIC worker log\n[sweep] {plan.run_id} -> completed\n")
    return SimpleNamespace(
        root=root,
        run=run,
        plan=plan,
        plans=plans,
        manifest=manifest,
        log=log,
        native=native,
        destination=tmp_path / "SYNTHETIC-archive",
    )


def perform(item):
    return archive.archive_run(item.manifest, item.plan.run_id, item.log, item.destination)


def test_archive_preserves_every_byte_log_and_source_and_is_self_verifying(evidence):
    before = {
        name: archive.sha256(path) for name, path in archive.regular_files(evidence.run).items()
    }
    receipt = perform(evidence)
    assert receipt["checks_passed"] and not receipt["original_files_removed"]
    assert receipt["run_file_count"] == len(before)
    assert receipt["archive_sha256"] == archive.sha256(evidence.destination / "run.tar.gz")
    assert not (evidence.destination / "run.partial.tar.gz").exists()
    assert not (evidence.destination / "archive_failure.json").exists()
    assert before == {
        name: archive.sha256(path) for name, path in archive.regular_files(evidence.run).items()
    }
    inside = archive.verify_archive(evidence.destination / "run.tar.gz")
    assert set(f"run/{name}" for name in before) <= set(inside["files"])
    assert inside["files"]["worker/stdout.log"]["sha256"] == archive.sha256(evidence.log)
    assert "source/uv.lock" in inside["files"]
    assert "source/src/commstudy/SYNTHETIC.py" in inside["files"]
    assert (
        receipt["embedded_manifest_sha256"]
        == archive.hashlib.sha256(archive.json_bytes(inside)).hexdigest()
    )
    # Independent archive verification needs no original paths to remain present.
    evidence.run.rename(evidence.run.with_name("SYNTHETIC-hidden-original"))
    assert archive.verify_archive(evidence.destination / "run.tar.gz") == inside


def test_no_overwrite_or_destination_inside_source(evidence):
    before = set(evidence.run.rglob("*"))
    with pytest.raises(ValueError, match="outside"):
        archive.archive_run(
            evidence.manifest, evidence.plan.run_id, evidence.log, evidence.run / "unsafe-archive"
        )
    assert set(evidence.run.rglob("*")) == before
    evidence.destination.mkdir()
    marker = evidence.destination / "user-file"
    marker.write_bytes(b"must stay intact")
    with pytest.raises(ValueError, match="overwrite"):
        perform(evidence)
    assert marker.read_bytes() == b"must stay intact"
    assert list(evidence.destination.iterdir()) == [marker]


@pytest.mark.parametrize("when", ["during_validation", "during_write", "new_file", "worker_log"])
def test_mutation_is_rejected_without_publishing_completion(evidence, monkeypatch, when):
    original_write, original_validate = archive.write_tar, archive.validate_inputs

    def validate(*args):
        result = original_validate(*args)
        if when == "during_validation":
            (evidence.run / "metrics.csv").write_bytes(b"synthetic concurrent mutation")
        return result

    def write(*args):
        original_write(*args)
        if when == "during_write":
            (evidence.run / "metrics.csv").write_bytes(b"synthetic concurrent mutation")
        elif when == "new_file":
            (evidence.run / "unexpected-file").write_text("synthetic mutation")
        elif when == "worker_log":
            with evidence.log.open("a") as stream:
                stream.write("synthetic concurrent mutation\n")

    monkeypatch.setattr(archive, "validate_inputs", validate)
    monkeypatch.setattr(archive, "write_tar", write)
    with pytest.raises(ValueError, match="changed"):
        perform(evidence)
    assert not (evidence.destination / "archive_manifest.json").exists()
    assert not (evidence.destination / "run.tar.gz").exists()
    assert (evidence.destination / "archive_failure.json").exists()
    if when != "during_validation":
        assert (evidence.destination / "run.partial.tar.gz").exists()


@pytest.mark.parametrize("corruption", ["payload", "trailer", "duplicate", "traversal"])
def test_readback_rejects_corrupt_bytes_and_unsafe_members(evidence, corruption):
    perform(evidence)
    path = evidence.destination / "run.tar.gz"
    if corruption == "trailer":
        path.write_bytes(path.read_bytes()[:-4])
    else:
        with tarfile.open(path, "r:gz") as tar:
            contents = [(copy.copy(member), tar.extractfile(member).read()) for member in tar]
        with tarfile.open(path, "w:gz") as tar:
            for member, payload in contents:
                if corruption == "payload" and member.name == "run/metrics.csv":
                    payload = b"X" + payload[1:]
                if corruption == "traversal" and member.name == "run/metrics.csv":
                    member.name = "../escaped"
                tar.addfile(member, io.BytesIO(payload))
            if corruption == "duplicate":
                member, payload = contents[0]
                tar.addfile(member, io.BytesIO(payload))
    with pytest.raises((ValueError, EOFError, gzip.BadGzipFile, tarfile.ReadError)):
        archive.verify_archive(path)


def test_failed_write_leaves_partial_and_no_completion_marker(evidence, monkeypatch):
    def fail(path, *args):
        path.write_bytes(b"SYNTHETIC incomplete gzip")
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(archive, "write_tar", fail)
    with pytest.raises(OSError, match="interrupted"):
        perform(evidence)
    assert (
        evidence.destination / "run.partial.tar.gz"
    ).read_bytes() == b"SYNTHETIC incomplete gzip"
    assert not (evidence.destination / "archive_manifest.json").exists()
    assert (evidence.destination / "archive_failure.json").exists()


def test_cleanup_failure_cannot_invalidate_verified_publication(evidence, monkeypatch):
    original = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.name in ("run.partial.tar.gz", "archive_manifest.json.pending"):
            raise PermissionError("SYNTHETIC post-publication cleanup error")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    receipt = perform(evidence)
    assert receipt["checks_passed"]
    assert (evidence.destination / "archive_manifest.json").is_file()
    assert archive.sha256(evidence.destination / "run.tar.gz") == receipt["archive_sha256"]
    assert not (evidence.destination / "archive_failure.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("frames", 594000),
        ("iterations", 99),
        ("seed", 10),
        ("source_sha256", "wrong"),
        ("status", "failed"),
        ("execution_purpose", "validation"),
    ],
)
def test_archive_scope_rejects_wrong_completion_identity(evidence, field, value):
    path = evidence.run / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    dump(path, metadata)
    with pytest.raises(ValueError, match="identity"):
        perform(evidence)
    assert not (evidence.destination / "archive_manifest.json").exists()


def test_symbolic_link_is_never_followed(evidence):
    (evidence.run / "linked-file").symlink_to(evidence.log)
    with pytest.raises(ValueError, match="Non-regular"):
        perform(evidence)
    assert evidence.log.exists()


def test_worker_log_must_belong_to_completed_run(evidence):
    evidence.log.write_text("SYNTHETIC unfinished worker")
    with pytest.raises(ValueError, match="completed run marker"):
        perform(evidence)


def test_boolean_checkpoint_schema_is_not_an_integer_version(evidence):
    path = evidence.run / "checkpoints/policy_state.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        perform(evidence)
    assert not (evidence.destination / "archive_manifest.json").exists()


def test_reviewed_inventory_reader_is_callable_without_policy_rollout():
    reader = archive.inventory_reader()
    assert callable(reader)
