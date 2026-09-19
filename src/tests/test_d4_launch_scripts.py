"""Artifact corruption regressions for the standalone D4 launch/audit scripts."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke_checkpoint(tmp_path, monkeypatch):
    module = script("validate_pcp_budget_smokes.py")
    run = tmp_path / "suite" / "identity"
    (run / "checkpoints").mkdir(parents=True)
    policies = {group: torch.nn.Linear(2, 2) for group in ("adversary", "agent")}
    state = {
        "schema_version": 1, "frames": 6000, "run_id": run.name,
        "suite_id": run.parent.name, "task": "vmas_predator_capture_prey", "algorithm": "mappo",
        "model": "pcp_comm_identity", "seed": 10,
        "group_policies": {group: policy.state_dict() for group, policy in policies.items()},
    }
    path = run / "checkpoints/policy_state.pt"
    torch.save(state, path)
    experiment = SimpleNamespace(group_policies=policies, close=lambda: None)
    monkeypatch.setattr(module, "load_frozen_experiment", lambda *a, **kw: experiment)
    return module, run, state, path


def test_smoke_checkpoint_requires_readable_exact_finite_reload(smoke_checkpoint, tmp_path):
    module, run, _, _ = smoke_checkpoint
    result = module.validate_policy_checkpoint(run, {"model": "pcp_comm_identity"}, tmp_path)
    assert result["strict_reload_verified"]
    assert result["finite_tensor_counts_by_group"] == {"adversary": 2, "agent": 2}


@pytest.mark.parametrize("corruption", ["seed", "frames", "boolean_schema", "missing_group",
                                      "nonfinite", "dtype", "wrong_weights", "encoding"])
def test_smoke_rejects_bad_checkpoint_even_if_metrics_and_metadata_pass(
    smoke_checkpoint, tmp_path, corruption,
):
    module, run, state, path = smoke_checkpoint
    if corruption == "seed":
        state["seed"] = 11
    elif corruption == "frames":
        state["frames"] = 0
    elif corruption == "boolean_schema":
        state["schema_version"] = True
    elif corruption == "missing_group":
        state["group_policies"].pop("agent")
    elif corruption == "nonfinite":
        state["group_policies"]["adversary"]["weight"] = torch.full((2, 2), float("nan"))
    elif corruption == "dtype":
        state["group_policies"]["adversary"]["weight"] = (
            state["group_policies"]["adversary"]["weight"].double()
        )
    elif corruption == "wrong_weights":
        state["group_policies"]["adversary"]["weight"] = torch.ones(2, 2) * 99
    if corruption == "encoding":
        path.write_bytes(b"corrupt checkpoint")
    else:
        torch.save(state, path)
    with pytest.raises((ValueError, pickle.UnpicklingError)):
        module.validate_policy_checkpoint(run, {"model": "pcp_comm_identity"}, tmp_path)


@pytest.fixture
def retained_checkpoints(tmp_path):
    module = script("evaluate_pcp_budget.py")
    run = tmp_path / "run"
    directory = run / "benchmarl" / "fixture" / "checkpoints"
    directory.mkdir(parents=True)
    for frames in range(120000, 600001, 120000):
        state = {
            "state": {"total_frames": frames, "n_iters_performed": frames // 6000},
            **{f"loss_{group}": {
                "actor_network_params.weight": torch.full((2, 2), float(frames)),
                "critic_network_params.weight": torch.full((3, 3), float(frames)),
                "actor_network_params.__batch_size": torch.Size([]),
                "actor_network_params.__device": None,
            } for group in ("adversary", "agent")},
        }
        torch.save(state, directory / f"checkpoint_{frames}.pt")
    managed = run / "checkpoints/policy_state.pt"
    managed.parent.mkdir()
    torch.save({"group_policies": {
        group: {"weight": torch.full((2, 2), 600000.0)} for group in ("adversary", "agent")
    }}, managed)
    return module, run, directory


def test_retention_inventory_preserves_all_frames_and_existing_diagnostics(retained_checkpoints):
    module, run, directory = retained_checkpoints
    diagnostics = directory.parent / "replay_diagnostics"
    diagnostics.mkdir()
    torch.save({"diagnostic_only": True, "not_resume": True}, diagnostics / "snapshot.pt")
    inventory = module.retained_checkpoint_inventory(run)
    expected_frames = list(range(120000, 600001, 120000))
    assert [row["frames"] for row in inventory["checkpoints"]] == expected_frames
    assert len(inventory["replay_diagnostics"]) == 1
    assert inventory["final_native_actor_matches_managed"]


@pytest.mark.parametrize("corruption", ["extra_frame", "missing_frame", "wrong_iterations",
                                      "empty_critic", "missing_group", "nonfinite", "shape_drift",
                                      "invalid_metadata"])
def test_retention_rejects_unusable_intermediate_checkpoints(retained_checkpoints, corruption):
    module, run, directory = retained_checkpoints
    path = directory / "checkpoint_360000.pt"
    state = torch.load(path, weights_only=False)
    if corruption == "extra_frame":
        torch.save(state, directory / "checkpoint_660000.pt")
    elif corruption == "missing_frame":
        path.unlink()
    else:
        if corruption == "wrong_iterations":
            state["state"]["n_iters_performed"] = 60.0
        elif corruption == "empty_critic":
            state["loss_agent"].pop("critic_network_params.weight")
        elif corruption == "missing_group":
            state.pop("loss_agent")
        elif corruption == "nonfinite":
            state["loss_adversary"]["actor_network_params.weight"] = torch.full(
                (2, 2), float("nan"),
            )
        elif corruption == "shape_drift":
            state["loss_adversary"]["actor_network_params.weight"] = torch.ones(3, 2)
        else:
            state["loss_agent"]["actor_network_params.__batch_size"] = ()
        torch.save(state, path)
    with pytest.raises(ValueError):
        module.retained_checkpoint_inventory(run)


@pytest.mark.parametrize("corruption", ["values", "dtype", "keys", "signed_zero"])
def test_final_native_actor_must_match_managed_evaluation_actor(retained_checkpoints, corruption):
    module, run, directory = retained_checkpoints
    managed = run / "checkpoints/policy_state.pt"
    saved = torch.load(managed, weights_only=False)
    actor = saved["group_policies"]["adversary"]
    if corruption == "values":
        actor["weight"][0, 0] += 1.0
    elif corruption == "dtype":
        actor["weight"] = actor["weight"].double()
    elif corruption == "keys":
        actor["incorrect_weight"] = actor.pop("weight")
    else:
        native_path = directory / "checkpoint_600000.pt"
        native = torch.load(native_path, weights_only=False)
        native["loss_adversary"]["actor_network_params.weight"][0, 0] = -0.0
        torch.save(native, native_path)
        actor["weight"][0, 0] = 0.0
    torch.save(saved, managed)
    with pytest.raises(ValueError, match="Final native and managed actor"):
        module.retained_checkpoint_inventory(run)
