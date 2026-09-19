"""Fixed protocol gates and artifact lifecycle; no simulator or scientific run."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def confirmation():
    import sys

    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts)
    try:
        yield importlib.import_module("run_mapdn_confirmation")
    finally:
        sys.path.remove(scripts)


@pytest.fixture
def bank():
    return [{"episode_id": f"test:{i}", "start_row": i * 1000, "seed": 300000 + i}
            for i in range(16)]


def outcome(bank, magnitude=0.0, violating=0, effort=1.0):
    episodes = []
    for entry in bank:
        transitions = []
        for step in range(239):
            transitions.append({
                "step": step, "terminated": False, "truncated": step == 238,
                "actions": [[0.0]] * 6,
                "diagnostics": {"valid": 1, "voltage_metrics_valid": 1, "feasible_step": 1,
                    "destroy": 0, "action_solver_failure": 0, "advance_solver_failure": 0,
                    "reward_row": entry["start_row"] + step,
                    "observation_row": entry["start_row"] + step + 1,
                    "voltage_violation_magnitude": magnitude,
                    "percentage_of_v_out_of_control": violating / 33,
                    "voltage_bus_count": 33, "voltage_violating_bus_count": violating,
                    "q_loss": effort},
            })
        episodes.append({**entry, "transitions": transitions, "transition_count": 239,
                         "complete": True})
    return {"bank": bank, "episodes": episodes, "complete_episodes": 16}


def test_gate_primary_requires_reduction_and_nonworse_frequency(confirmation, bank):
    initial = outcome(bank, magnitude=0.1, violating=2)
    good = outcome(bank, magnitude=0.094, violating=1)
    gate = confirmation.domain_gate(initial, good, bank, 33)
    assert gate["passed"] and gate["branch"] == "voltage_reduction"
    assert gate["initial"]["violated_bus_count"] == 2 * 3824
    assert gate["initial"]["bus_transition_count"] == 33 * 3824
    assert gate["initial"]["action_diagnostics"]["action_scalar_denominator"] == 6 * 3824
    for final in (outcome(bank, magnitude=0.096, violating=2),
                  outcome(bank, magnitude=0.01, violating=3)):
        assert not confirmation.domain_gate(initial, final, bank, 33)["passed"]


def test_secondary_requires_positive_effort_and_zero_voltage_violations(confirmation, bank):
    initial = outcome(bank)
    assert confirmation.domain_gate(initial, outcome(bank, effort=0.94), bank, 33)["passed"]
    for final in (outcome(bank, effort=0.96), outcome(bank, magnitude=1e-8, effort=0.5),
                  outcome(bank, violating=1, effort=0.5)):
        assert not confirmation.domain_gate(initial, final, bank, 33)["passed"]
    assert not confirmation.domain_gate(
        outcome(bank, effort=0), outcome(bank, effort=0), bank, 33)["passed"]


@pytest.mark.parametrize("mutation", ["short", "failure", "stale", "row", "seed", "count", "nan"])
def test_incomplete_or_invalid_bank_cannot_improve_by_changing_denominator(
    confirmation, bank, mutation
):
    initial, final = outcome(bank), outcome(bank, effort=0.5)
    first = final["episodes"][0]
    diagnostics = first["transitions"][0]["diagnostics"]
    if mutation == "short":
        first["transitions"].pop()
    elif mutation == "failure":
        diagnostics["destroy"] = 1
    elif mutation == "stale":
        diagnostics["voltage_metrics_valid"] = 0
    elif mutation == "row":
        diagnostics["observation_row"] += 1
    elif mutation == "seed":
        first["seed"] += 1
    elif mutation == "count":
        diagnostics["voltage_violating_bus_count"] = 0.5
    else:
        diagnostics["q_loss"] = float("nan")
    result = confirmation.domain_gate(initial, final, bank, 33)
    assert not result["passed"] and not result["final"]["valid"]


def test_fixed_bank_uses_integer_positions_and_nonoverlapping_windows(confirmation):
    manifest = {"splits": {"test": {"starts": list(range(20001))}}}
    bank = confirmation._fixed_bank(manifest, "test", 300000)
    assert [entry["start_row"] for entry in bank] == [i * 20000 // 15 for i in range(16)]
    manifest["splits"]["test"]["starts"] = list(range(1000))
    with pytest.raises(ValueError, match="overlap"):
        confirmation._fixed_bank(manifest, "test", 300000)


def test_resolved_spec_explicitly_binds_plan_optimizer_and_checkpoint_settings(confirmation):
    root = Path(__file__).resolve().parents[2]
    config = confirmation.asdict(confirmation.TaskConfig(max_steps=239, measurement_noise=False))
    spec = confirmation._spec(root, config, 30)
    expected = {"max_n_frames": 8192, "on_policy_collected_frames_per_batch": 256,
                "on_policy_minibatch_size": 64, "on_policy_n_minibatch_iters": 4,
                "adam_eps": 1e-6, "lr": 5e-5, "gamma": 0.99, "clip_grad_val": 5.0,
                "clip_grad_norm": True, "evaluation": False, "checkpoint_interval": 4096,
                "keep_checkpoints_num": None, "checkpoint_at_end": True}
    assert all(spec.experiment[key] == value for key, value in expected.items())
    assert "exclude_buffer_from_checkpoint" in spec.experiment
    assert spec.algorithm_config["params"]["entropy_coef"] == 0
    assert spec.model_config["params"]["hidden_dim"] == 128
    assert spec.critic_model["params"]["num_cells"] == [128, 128]


def test_reviewed_digest_rejects_rehashed_preparation_before_launch(confirmation, tmp_path):
    packet = {"status": "PREPARED"}
    packet["sha256"] = confirmation._canonical_sha(packet)
    reviewed = packet["sha256"]
    packet["status"] = "modified"
    packet["sha256"] = confirmation._canonical_sha(packet)
    confirmation._write_new_json(tmp_path / "preparation.json", packet)
    with pytest.raises(ValueError, match="content hash"):
        confirmation.verify_preparation(tmp_path, tmp_path, reviewed)
    assert not (tmp_path / "launch.json").exists()


def test_allocation_finishes_both_trainings_before_any_test_and_rejects_rerun(
    confirmation, tmp_path, monkeypatch
):
    events = []
    packet = {"sha256": "fixed", "sources": {}, "runtime": {}}
    monkeypatch.setattr(confirmation, "verify_preparation", lambda *args: packet)
    monkeypatch.setattr(confirmation, "_sources", lambda *args: {})
    monkeypatch.setattr(confirmation, "_runtime", lambda: {})
    monkeypatch.setattr(confirmation, "_archive", lambda *args: {"readback_verified": True})
    def train(*args):
        events.append(f"train:{args[-1]}")
        return {"seed": args[-1]}
    def measure(output, row, banks, packet):
        events.append(f"test:{row['seed']}")
        return {"seed": row["seed"], "passed": True}
    monkeypatch.setattr(confirmation, "_train_one", train)
    monkeypatch.setattr(confirmation, "_measure_seed", measure)
    confirmation._write_new_json(tmp_path / "banks.json", {})
    assert confirmation.execute(tmp_path, tmp_path, "fixed") == 0
    assert events == ["train:30", "train:31", "test:30", "test:31"]
    saved = (tmp_path / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        confirmation.execute(tmp_path, tmp_path, "fixed")
    assert (tmp_path / "report.json").read_bytes() == saved


def test_source_change_before_second_seed_stops_without_any_test(
    confirmation, tmp_path, monkeypatch
):
    calls, events = [], []
    packet = {"sha256": "fixed", "sources": {}, "runtime": {}}
    def verify(*args):
        calls.append(None)
        if len(calls) == 3:
            raise ValueError("source drift")
        return packet
    monkeypatch.setattr(confirmation, "verify_preparation", verify)
    monkeypatch.setattr(confirmation, "_sources", lambda *args: {})
    monkeypatch.setattr(confirmation, "_train_one", lambda *args: events.append("train") or {})
    monkeypatch.setattr(confirmation, "_measure_seed", lambda *args: pytest.fail("test opened"))
    assert confirmation.execute(tmp_path, tmp_path, "fixed") == 1
    assert events == ["train"]
    report = confirmation.read_json(tmp_path / "report.json")
    assert report["verdict"] == "NOT_CONFIRMED" and report["failure_stage"] == "training"


def test_archive_exact_inventory_and_symlink_rejection(confirmation, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty").mkdir()
    (source / "actor.bin").write_bytes(b"checkpoint fixture")
    receipt = confirmation._archive(source, tmp_path / "archive.tar.gz")
    assert receipt["readback_verified"]
    assert receipt["inventory"]["empty"] == {"kind": "directory"}
    (source / "link").symlink_to(source / "actor.bin")
    with pytest.raises(ValueError, match="symlinks"):
        confirmation._archive(source, tmp_path / "bad.tar.gz")


def test_native_checkpoint_metadata_preserved_and_nonfinite_tensors_rejected(confirmation):
    expected = {"actor.weight": torch.ones(2), "critic.weight": torch.ones(3),
                "actor.__batch_size": torch.Size([]), "actor.__device": None}
    losses = {"agents": SimpleNamespace(state_dict=lambda: expected)}
    saved = {"state": {"total_frames": 64}, "loss_agents": copy.deepcopy(expected)}
    confirmation._validate_native_state(saved, losses, 64)
    saved["loss_agents"]["actor.__batch_size"] = []
    with pytest.raises(ValueError, match="metadata"):
        confirmation._validate_native_state(saved, losses, 64)
    saved["loss_agents"] = copy.deepcopy(expected)
    saved["loss_agents"]["critic.weight"][0] = 2.0
    with pytest.raises(ValueError, match="final model"):
        confirmation._validate_native_state(saved, losses, 64, require_equal=True)
    saved["loss_agents"]["critic.weight"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        confirmation._validate_native_state(saved, losses, 64)


def test_existing_real_smoke_native_checkpoint_metadata_supported(confirmation):
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "results" / "mapdn_real_smoke_final_source_20260909").glob(
        "runs/**/checkpoints/checkpoint_64.pt"))
    if not paths:
        pytest.skip("Optional retained real smoke checkpoint is unavailable")
    saved = torch.load(paths[0], map_location="cpu", weights_only=False)
    expected = saved["loss_agents"]
    losses = {"agents": SimpleNamespace(state_dict=lambda: expected)}
    confirmation._validate_native_state(saved, losses, 64)


def test_saturation_uses_actual_point_eight_physical_bound(confirmation, bank):
    summary = outcome(bank)
    summary["episodes"][0]["transitions"][0]["actions"] = [[0.8], [-0.8], [0.5], [0], [0], [0]]
    result = confirmation.bank_metrics(summary, bank, 33, 6)["action_diagnostics"]
    assert result["saturated_action_scalars"] == 2
    assert result["action_scalar_denominator"] == 3824 * 6
    assert result["physical_bound_saturation_fraction"] == 2 / (3824 * 6)
