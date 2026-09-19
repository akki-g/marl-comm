"""D4 named CLI presets reproduce the protocol's fully resolved scientific rows.

These checks construct policies but never collect or optimize training data.
Only the external approval check is isolated: approval artifacts are deployment
evidence and are tested separately by the protocol gate suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from commstudy.experiments.config import (
    load_experiment_spec,
    resolved_experiment_dict,
    scientific_config_sha256,
)
from commstudy.experiments.protocols import expected_model_contract, load_protocol
from commstudy.experiments.sweeps import expand_suite_config, validate_plan


ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "pcp_comm_identity",
    "pcp_local_capacity",
    "pcp_broadcast_k0",
    "pcp_broadcast_k3",
)
VISIBILITIES = ("radius1", "global")


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(
        ROOT / "configs/protocols/pcp_visibility_budget_cpu_v1_frozen.yaml"
    )


@pytest.fixture(scope="module")
def planned_rows():
    # Use the real suite expander, including its actual built model contracts;
    # do not write a manifest or mutate the existing managed suite.
    suite = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/sweeps/pcp_visibility_budget_cpu_v1.yaml"),
        resolve=True,
    )
    rows = expand_suite_config(suite, repo_root=ROOT)
    assert len(rows) == 40
    assert {(row.model, row.ablation_value, row.seed) for row in rows} == {
        (model, visibility, seed)
        for model in MODELS
        for visibility in VISIBILITIES
        for seed in range(20, 25)
    }
    return rows


@pytest.mark.parametrize("model", MODELS[1:])
def test_named_preset_is_exact_declared_method_without_inline_model_override(
    model, protocol, config_root,
):
    document = OmegaConf.to_container(
        OmegaConf.load(config_root / "models" / f"{model}.yaml"), resolve=True
    )
    assert document == protocol["methods"][model]
    named = load_experiment_spec(config_root, [
        "algorithm=mappo",
        "task=vmas_predator_capture_prey",
        f"model={model}",
        "critic_model=pcp_critic",
    ])
    assert named.model_config == protocol["methods"][model]


@pytest.mark.parametrize("visibility", VISIBILITIES)
@pytest.mark.parametrize("model", MODELS)
def test_named_cli_plan_resolution_preserves_all_seeds_and_built_contract(
    model, visibility, planned_rows, config_root, monkeypatch,
):
    approvals_checked = []

    def isolated_approval_check(spec, binding, *, repo_root, check_runtime):
        assert repo_root == ROOT
        assert check_runtime is False
        assert binding["model"] == spec.model
        assert binding["seed"] == spec.seed
        approvals_checked.append(spec.seed)

    monkeypatch.setattr(
        "commstudy.experiments.protocols.validate_protocol_launch",
        isolated_approval_check,
    )
    rows = [row for row in planned_rows
            if (row.model, row.ablation_value) == (model, visibility)]
    assert {row.seed for row in rows} == set(range(20, 25))
    for row in rows:
        # This is the worker's real named-CLI load, including the prerequisite
        # preset file lookup even when resolved overrides contain model_config.
        named = load_experiment_spec(config_root, row.overrides)
        validated = validate_plan(
            row, config_root=config_root, repo_root=ROOT, check_runtime=False
        )
        assert scientific_config_sha256(named) == row.scientific_hash
        assert resolved_experiment_dict(named) == row.resolved_spec
        assert resolved_experiment_dict(validated) == row.resolved_spec
    assert sorted(approvals_checked) == list(range(20, 25))
    # Build from the named loader itself, not from the saved snapshot returned
    # by validate_plan. The protocol planner builds the reference independently.
    actual_contract = expected_model_contract(named)
    assert all(actual_contract == row.model_contract for row in rows)
    counts = actual_contract["parameter_counts"]
    assert counts["group/adversary/actor_total"] == (
        19332 if model == "pcp_comm_identity" else 27524
    )
