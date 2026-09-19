"""Strict retention-only cohort comparisons, using synthetic artifact corruption."""

from __future__ import annotations

import csv
import importlib.util
import json

import pytest
import torch
import yaml

from commstudy.experiments.config import resolved_experiment_dict, scientific_config_sha256
from commstudy.experiments.protocols import content_sha256, load_protocol, protocol_spec


@pytest.fixture
def retention_module(config_root):
    path = config_root.parent / "scripts/compare_pcp_retention_confirmation.py"
    spec = importlib.util.spec_from_file_location("retention_comparator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_checkpoint(run, frame, corruption=None):
    loss = {
        "actor_network_params.weight": torch.tensor([1.0, -0.0]),
        "critic_network_params.weight": torch.tensor([2.0, 3.0]),
        **{
            f"{network}_network_params.__batch_size": torch.Size([])
            for network in ("actor", "critic")
        },
        **{f"{network}_network_params.__device": None for network in ("actor", "critic")},
    }
    payload = {
        "state": {"total_frames": frame, "n_iters_performed": frame // 6000},
        "loss_adversary": loss,
        "loss_agent": {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in loss.items()
        },
    }
    if corruption == "value":
        payload["loss_agent"]["critic_network_params.weight"][0] += 0.01
    elif corruption == "dtype":
        payload["loss_agent"]["critic_network_params.weight"] = payload["loss_agent"][
            "critic_network_params.weight"
        ].double()
    elif corruption == "signed_zero":
        payload["loss_agent"]["actor_network_params.weight"][1] = 0.0
    path = run / f"benchmarl/native/checkpoints/checkpoint_{frame}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


@pytest.fixture
def cohorts(retention_module, config_root, tmp_path, monkeypatch):
    # The unit fixture binds a declared source token. Actual CLI execution must
    # independently hash the frozen copy, and is tested by its fail-closed path.
    monkeypatch.setattr(
        retention_module, "source_fingerprint", lambda root: retention_module.SOURCE
    )
    helper = retention_module.legacy_module()
    suites = [tmp_path / f"v{version}" for version in (1, 2, 3)]
    for version, suite in enumerate(suites, 1):
        protocol = load_protocol(config_root / f"protocols/pcp_corrected_cpu_v{version}.yaml")
        for seed in (10, 11):
            run = suite / f"seed{seed}"
            run.mkdir(parents=True)
            spec = protocol_spec(protocol, model="pcp_comm_identity", seed=seed)
            (run / "resolved_config.yaml").write_text(
                yaml.safe_dump({"commstudy": resolved_experiment_dict(spec)})
            )
            source = retention_module.ORIGINAL_SOURCE if version == 1 else retention_module.SOURCE
            status = "failed" if version == 1 else "completed"
            metadata = {
                "seed": seed,
                "model": "pcp_comm_identity",
                "status": status,
                "frames": 564000 if version == 1 else 600000,
                "source_sha256": source,
                "scientific_config_sha256": scientific_config_sha256(spec),
                "protocol": {
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": content_sha256(protocol),
                    "source_sha256": source,
                    "stage": "confirmation",
                },
                "versions": {
                    "python": protocol["runtime"]["python"],
                    **protocol["runtime"]["packages"],
                },
                "runtime": {
                    "sampling_device": "cpu",
                    "train_device": "cpu",
                    "buffer_device": "cpu",
                    "torch_num_threads": 1,
                },
            }
            (run / "metadata.json").write_text(json.dumps(metadata))
            (run / "status.json").write_text(json.dumps({"status": status}))
            with (run / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(helper.METRIC_COLUMNS))
                writer.writeheader()
                for frame in range(6000, (558000 if version == 1 else 600000) + 1, 6000):
                    for group in retention_module.GROUPS:
                        names = ["return", "wall_time_seconds", *sorted(helper.REPLAY_METRICS)]
                        for name in names:
                            value = 0 if name in helper.REPLAY_METRICS else 1
                            writer.writerow(
                                {
                                    "timestamp": f"version{version}",
                                    "frames": frame,
                                    "iteration": frame // 6000 - 1,
                                    "phase": "collection",
                                    "group": group,
                                    "metric": name,
                                    "sample": "",
                                    "value": value,
                                }
                            )
            for frame in (
                (480000,)
                if version == 1
                else (600000,)
                if version == 2
                else retention_module.CHECKPOINT_FRAMES
            ):
                _save_checkpoint(run, frame)
            if version == 3:
                (run / "checkpoints").mkdir()
                (run / "checkpoints/policy_state.pt").write_bytes(
                    b"separately audited final policy"
                )
                folder = run / "benchmarl/native/replay_diagnostics"
                folder.mkdir()
                (folder / "frame000564000_adversary_first_fallback.pt").write_bytes(
                    b"separate strict replay audit required"
                )
    return suites, config_root / "protocols/pcp_corrected_cpu_v3.yaml"


def test_retention_only_comparison_passes_with_all_required_durable_evidence(
    retention_module, cohorts
):
    suites, protocol = cohorts
    report = retention_module.compare(*suites, protocol)
    assert report["checks_passed"] and report["approval_decision"] == "not_issued"
    assert report["v2_full_exclusions"] == ["timestamp column", "wall_time_seconds metric"]
    for run in report["runs"]:
        assert len(run["retained_artifact_manifest"]) == 7
        assert run["v1_vs_v3_checkpoint_480k"]["tensor_count"] == 4
        assert run["artifact_hashes_before"] == run["artifact_hashes_after"]


@pytest.mark.parametrize(
    "frame,corruption",
    [(480000, "value"), (600000, "value"), (480000, "dtype"), (600000, "signed_zero")],
)
def test_retention_comparison_rejects_changed_loss_tensor_bytes(
    retention_module, cohorts, frame, corruption
):
    suites, protocol = cohorts
    _save_checkpoint(suites[2] / "seed10", frame, corruption)
    report = retention_module.compare(*suites, protocol)
    assert not report["checks_passed"]


@pytest.mark.parametrize(
    "corruption", ["missing480k", "missing120k", "keep", "gamma", "protocol", "runtime", "source"]
)
def test_retention_gate_rejects_missing_evidence_or_undeclared_settings(
    retention_module, cohorts, corruption
):
    suites, protocol = cohorts
    run = suites[2] / "seed10"
    if corruption.startswith("missing"):
        frame = 480000 if corruption == "missing480k" else 120000
        (run / f"benchmarl/native/checkpoints/checkpoint_{frame}.pt").unlink()
    elif corruption in ("keep", "gamma"):
        path = run / "resolved_config.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["commstudy"]["experiment"][
            "keep_checkpoints_num" if corruption == "keep" else "gamma"
        ] = 2 if corruption == "keep" else 0.9
        path.write_text(yaml.safe_dump(doc))
    else:
        path = run / "metadata.json"
        doc = json.loads(path.read_text())
        if corruption == "protocol":
            doc["protocol"]["protocol_sha256"] = "a" * 64
        elif corruption == "runtime":
            doc["runtime"]["torch_num_threads"] = 2
        else:
            doc["source_sha256"] = "a" * 64
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        retention_module.compare(*suites, protocol)


@pytest.mark.parametrize(
    "metric,frame",
    [("return", 6000), ("return", 600000), ("replay_log_prob_max_abs_error", 564000)],
)
def test_full_retention_comparison_includes_late_and_replay_metrics(
    retention_module, cohorts, metric, frame
):
    suites, protocol = cohorts
    path = suites[2] / "seed10/metrics.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    selected = next(row for row in rows if row["metric"] == metric and int(row["frames"]) == frame)
    selected["value"] = "0.01"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    assert not retention_module.compare(*suites, protocol)["checks_passed"]


def test_retention_cli_fails_closed_outside_frozen_source(
    retention_module, cohorts, monkeypatch, tmp_path
):
    suites, protocol = cohorts
    monkeypatch.setattr(retention_module, "source_fingerprint", lambda root: "a" * 64)
    out = tmp_path / "blocked.json"
    assert (
        retention_module.main(
            [
                "--original-suite",
                str(suites[0]),
                "--repaired-suite",
                str(suites[1]),
                "--retained-suite",
                str(suites[2]),
                "--protocol",
                str(protocol),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    assert "frozen" in json.loads(out.read_text())["issues"][0]


def test_cpu_v3_changes_only_retention_in_the_resolved_protocol(config_root):
    old = load_protocol(config_root / "protocols/pcp_corrected_cpu_v2.yaml")
    new = load_protocol(config_root / "protocols/pcp_corrected_cpu_v3.yaml")
    old_base, new_base = old["base_spec"], new["base_spec"]
    assert old_base["experiment"]["keep_checkpoints_num"] == 2
    assert new_base["experiment"]["keep_checkpoints_num"] is None
    new_base["experiment"]["keep_checkpoints_num"] = 2
    assert old_base == new_base
    for key in ("runtime", "contracts", "selection", "methods", "factors"):
        assert old[key] == new[key]


@pytest.mark.parametrize("case", ["changed_auditor", "wrong_protocol", "missing_seeds"])
def test_snapshot_wrapper_rejects_changed_auditor_or_wrong_scope(
    config_root, tmp_path, monkeypatch, case
):
    path = config_root.parent / "scripts/audit_pcp_retention_snapshot.py"
    spec = importlib.util.spec_from_file_location("retention_snapshot_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "source_fingerprint", lambda root: module.SOURCE)
    if case == "changed_auditor":
        monkeypatch.setattr(module, "AUDITOR_SHA", "a" * 64)
    version = 2 if case == "wrong_protocol" else 3
    out = tmp_path / "review.json"
    assert (
        module.main(
            [
                str(tmp_path / "empty"),
                "--previous-suite",
                str(tmp_path / "old"),
                "--protocol",
                str(config_root / f"protocols/pcp_corrected_cpu_v{version}.yaml"),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    report = json.loads(out.read_text())
    assert not report["checks_passed"] and report["approval_decision"] == "not_issued"
    assert report["issues"]


@pytest.mark.parametrize("field,value", [("__batch_size", ()), ("__device", "cpu")])
def test_checkpoint_comparison_rejects_identically_malformed_metadata(
    retention_module, tmp_path, field, value
):
    left, right = tmp_path / "left", tmp_path / "right"
    for run in (left, right):
        path = _save_checkpoint(run, 480000)
        payload = torch.load(path, weights_only=False)
        payload["loss_agent"][f"actor_network_params.{field}"] = value
        torch.save(payload, path)
    with pytest.raises(ValueError, match="metadata type/value"):
        retention_module.compare_checkpoint(left, right, 480000)
