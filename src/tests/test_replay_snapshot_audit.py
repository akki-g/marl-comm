"""Standalone offline snapshot audit checks; no training or approval issuance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from commstudy.experiments import build_experiment, load_experiment_spec
from commstudy.experiments.health import check_behavior_replay


@pytest.fixture
def audit_module(config_root):
    spec = importlib.util.spec_from_file_location(
        "replay_snapshot_audit", config_root.parent / "scripts/audit_pcp_replay_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot_case(config_root, tmp_path):
    fixture = torch.load(
        Path(__file__).parent / "fixtures/pcp_replay_roundoff_seed010.pt", weights_only=True
    )
    spec = load_experiment_spec(
        config_root,
        [
            "task=vmas_predator_capture_prey",
            "model=pcp_comm_identity",
            "critic_model=pcp_critic",
            "experiment.max_n_frames=6000",
            "experiment.evaluation=false",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
            f"experiment.save_folder={tmp_path}",
        ],
    )
    experiment = build_experiment(spec)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = experiment.group_policies["adversary"]
        policy.load_state_dict(fixture["actor_state"], strict=True)
        batch = TensorDict(
            {"adversary": {key: fixture[key] for key in ("observation", "action", "log_prob")}},
            [10, 600],
            names=[None, "time"],
        )
        # Recreate collection-width distribution parameters, as the real saved
        # behavior batch contains loc/scale before checking its probabilities.
        with torch.no_grad():
            chunks = []
            for step in batch.unbind(1):
                replay = step.clone()
                policy.get_dist(replay)
                chunks.append(replay)
            batch = torch.stack(chunks, dim=1).refine_names(None, "time")
        details, capture = {}, {}
        metrics = check_behavior_replay(
            policy, batch, "adversary", diagnostics=details, capture=capture
        )
        if not metrics["replay_collection_shape_fallback_count"]:
            pytest.skip("This backend does not reproduce the macOS/ARM numerical fixture.")
        yield experiment, {"batch": batch, "details": details, "replay_tensors": capture}
    finally:
        torch.set_num_threads(prior_threads)
        experiment.close()


def test_offline_snapshot_replay_matches_capture_without_model_or_rng_change(
    audit_module, snapshot_case
):
    experiment, payload = snapshot_case
    result = audit_module.replay_snapshot(experiment, payload)
    assert result["whole_batch"]["violation_count"] > 0
    assert result["collection_shape"]["bitwise_equal"]
    assert result["stored_distribution"]["allclose"]
    assert all(
        field["bitwise_equal"] for field in result["stored_distribution_parameters"].values()
    )
    assert result["model_unchanged"] and result["rng_unchanged"]
    assert all(
        field["bitwise_equal"]
        for mode in result["captured_tensors"].values()
        for field in mode.values()
    )


@pytest.mark.parametrize(
    "corruption", ["capture", "captured_dtype", "stored_loc", "stored_scale", "weights", "names"]
)
def test_offline_snapshot_rejects_corrupted_capture_weights_or_shape(
    audit_module, snapshot_case, corruption
):
    experiment, payload = snapshot_case
    if corruption == "capture":
        payload["replay_tensors"]["batched"]["loc"].add_(0.1)
    elif corruption == "captured_dtype":
        payload["replay_tensors"]["batched"]["loc"] = payload["replay_tensors"]["batched"][
            "loc"
        ].double()
    elif corruption.startswith("stored_"):
        payload["batch"]["adversary", corruption.removeprefix("stored_")].add_(0.1)
    elif corruption == "weights":
        with torch.no_grad():
            next(experiment.group_policies["adversary"].parameters()).add_(0.1)
    else:
        payload["batch"].rename_(None, None)
    with pytest.raises(ValueError):
        audit_module.replay_snapshot(experiment, payload)


def _observations(module, iteration=713):
    observations = {}
    for group in ("adversary", "agent"):
        for frame in range(6000, module.FRAME, 6000):
            for name in (
                "replay_collection_shape_fallback_count",
                "replay_collection_shape_fallback_slices",
            ):
                observations[("collection", group, name, frame, "")] = (0, 0)
    for name in module.REPLAY_FIELDS:
        value = 600 if name == "replay_collection_shape_fallback_slices" else 1
        observations[("collection", "adversary", name, module.FRAME, "")] = (value, iteration)
    return observations


def test_recorded_collection_iteration_is_read_from_csv_records_not_inferred(audit_module):
    observations = _observations(audit_module)
    bound = audit_module._metric_binding(observations, ["adversary", "agent"])
    assert bound["recorded_collection_iteration"] == 713


@pytest.mark.parametrize("corruption", ["earlier_predator", "earlier_prey", "missing", "iteration"])
def test_snapshot_binding_rejects_earlier_fallback_or_bad_csv_records(audit_module, corruption):
    observations = _observations(audit_module)
    metric = "replay_collection_shape_fallback_count"
    if corruption.startswith("earlier_"):
        group = "agent" if corruption == "earlier_prey" else "adversary"
        observations[("collection", group, metric, 6000, "")] = (1, 0)
    elif corruption == "missing":
        del observations[("collection", "agent", metric, 6000, "")]
    else:
        observations[("collection", "adversary", metric, audit_module.FRAME, "")] = (1, 999)
    with pytest.raises(ValueError):
        audit_module._metric_binding(observations, ["adversary", "agent"])


def test_missing_snapshot_suite_writes_blocked_report_and_provenance_sidecar(
    audit_module, tmp_path
):
    out = tmp_path / "review.json"
    result = audit_module.main(
        [str(tmp_path / "new"), "--previous-suite", str(tmp_path / "old"), "--out", str(out)]
    )
    assert result == 2
    report = json.loads(out.read_text())
    assert not report["checks_passed"] and report["approval_decision"] == "not_issued"
    assert json.loads(out.with_suffix(".provenance.json").read_text()) == {}


@pytest.mark.parametrize("corruption", [None, "protocol", "model", "configuration"])
def test_previous_failure_must_belong_to_the_matching_cpu_v1_scientific_case(
    audit_module, tmp_path, corruption
):
    run = tmp_path / "old_seed10"
    run.mkdir()
    current = {"seed": 10, "scientific_config_sha256": "a" * 64}
    prior = {
        **current,
        "model": "pcp_comm_identity",
        "protocol": {"protocol_id": "pcp_corrected_cpu_v1"},
    }
    if corruption == "protocol":
        prior["protocol"]["protocol_id"] = "unrelated_protocol"
    elif corruption == "model":
        prior["model"] = "pcp_comm_attention"
    elif corruption == "configuration":
        prior["scientific_config_sha256"] = "b" * 64
    (run / "metadata.json").write_text(json.dumps(prior))
    if corruption is None:
        matched, metadata = audit_module._previous_run(tmp_path, current)
        assert matched == run and metadata == prior
    else:
        with pytest.raises(ValueError, match="same scientific CPUv1"):
            audit_module._previous_run(tmp_path, current)


@pytest.mark.parametrize("corruption", [None, "dtype", "missing_group"])
def test_all_snapshot_policy_groups_load_exact_tensor_bytes(audit_module, corruption):
    experiment = SimpleNamespace(
        group_policies={group: torch.nn.Linear(2, 2) for group in ("adversary", "agent")}
    )
    states = {
        group: {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
        for group, policy in experiment.group_policies.items()
    }
    if corruption == "dtype":
        states["agent"]["weight"] = states["agent"]["weight"].double()
    elif corruption == "missing_group":
        del states["agent"]
    if corruption is None:
        hashes = audit_module._load_snapshot_policies(experiment, states)
        assert set(hashes) == set(experiment.group_policies)
        assert hashes["agent"]["weight"] == audit_module._tensor_hash(states["agent"]["weight"])
    else:
        with pytest.raises(ValueError):
            audit_module._load_snapshot_policies(experiment, states)


@pytest.mark.parametrize("corruption", [None, "batch_size_type", "batch_size_value", "device"])
def test_real_pcp_prey_tensordict_metadata_matches_rebuilt_policy_exactly(
    audit_module, snapshot_case, corruption
):
    experiment, _ = snapshot_case
    states = {group: policy.state_dict() for group, policy in experiment.group_policies.items()}
    batch_key = next(name for name in states["agent"] if name.endswith(".params.__batch_size"))
    device_key = next(name for name in states["agent"] if name.endswith(".params.__device"))
    assert type(states["agent"][batch_key]) is torch.Size
    assert states["agent"][device_key] is None
    if corruption == "batch_size_type":
        states["agent"][batch_key] = tuple(states["agent"][batch_key])
    elif corruption == "batch_size_value":
        states["agent"][batch_key] = torch.Size([99])
    elif corruption == "device":
        states["agent"][device_key] = torch.device("cpu")
    if corruption is None:
        hashes = audit_module._load_snapshot_policies(experiment, states)
        assert hashes["agent"][batch_key] == audit_module._policy_metadata_hash(
            batch_key, torch.Size([])
        )
    else:
        with pytest.raises(ValueError, match="metadata"):
            audit_module._load_snapshot_policies(experiment, states)
