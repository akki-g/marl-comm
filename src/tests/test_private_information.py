"""Scientific attribution and estimand checks without simulator rollouts."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from commstudy.analysis.private_information import (
    _pcp_setup, default_plan, finite_action_summary, json_digest,
    observation_comparison, source_inventory, verify_plan,
)


def test_pair_with_conflicting_actions_has_positive_finite_grid_information_value():
    # An equally likely state requires either action; the best common plan earns 2.
    result = finite_action_summary([[4, 0, 1], [0, 4, 1]], attributable=True)
    assert result["mean_state_best_utility"] == 4
    assert result["best_common_action_mean_utility"] == 2
    assert result["private_information_value"] == 2
    assert result["maximizer_sets"] == [[0], [1]]
    assert result["strict_preference_conflict"]


def test_overlapping_maximizers_do_not_create_false_preference_reversal():
    result = finite_action_summary([[4, 4, 0], [0, 4, 4]], attributable=True)
    assert result["maximizer_sets"] == [[0, 1], [1, 2]]
    assert not result["strict_preference_conflict"]
    assert result["private_information_value"] == 0


def test_observed_state_difference_cannot_be_attributed_to_private_information():
    result = finite_action_summary([[4, 0], [0, 4]], attributable=False)
    assert result["conditional_state_information_value"] == 2
    assert result["private_information_value"] is None
    assert not result["private_information_attribution_valid"]


def test_nonfinite_or_missing_actions_invalidate_comparison_without_changing_denominator():
    result = finite_action_summary([[4, np.nan], [0, 4]], attributable=True)
    assert result == {"complete": False, "private_information_attribution_valid": False}
    with pytest.raises(ValueError, match="two-state"):
        finite_action_summary([[4, 0]], attributable=True)


def test_exact_observation_gate_rejects_arbitrarily_small_coupling():
    result = observation_comparison([[1, 0], [1, 1e-15]], [[0, 1], [0, 2]])
    assert not result["recipient_exact_equal"]
    assert result["recipient_linf_distance"] == 1e-15
    assert not result["donor_exact_equal"]
    assert result["donor_linf_distance"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        observation_comparison([[1, np.nan], [1, 0]], [[0, 1], [0, 2]])


def test_control_observations_and_no_signal_remain_distinct_gates():
    result = observation_comparison([[1, 2], [1, 2]], [[2, 3], [2, 3]])
    assert result["recipient_exact_equal"]
    assert result["donor_exact_equal"]  # No private signal is present.


def test_plan_changes_cannot_be_relabelled_as_frozen_version(tmp_path):
    plan = default_plan()
    plan["pcp"]["horizon"] += 1
    plan["sha256"] = json_digest(plan)
    with pytest.raises(ValueError, match="prospective version"):
        verify_plan(plan, tmp_path)
    plan["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_plan(plan, tmp_path)


def test_design_is_serializable_and_finite_action_grid_is_bounded():
    plan = default_plan()
    assert json.loads(json.dumps(plan, allow_nan=False)) == plan
    actions = np.asarray(plan["pcp"]["actions"])
    assert actions.shape == (9, 2)
    assert np.max(np.abs(actions)) <= 1
    assert plan["mapdn"]["actions"] == [-0.8, -0.4, 0.0, 0.4, 0.8]
    assert len(plan["pcp"]["seeds"]) == 2


def test_runtime_receipt_survives_json_roundtrip():
    root = Path(__file__).resolve().parents[2]
    functions = runpy.run_path(str(root / "scripts/diagnose_private_information.py"))
    identity = functions["runtime_identity"]()
    assert identity == json.loads(json.dumps(identity, allow_nan=False))


def test_source_inventory_includes_native_task_factory_yaml_and_vendor_metadata():
    inventory = source_inventory(Path(__file__).resolve().parents[2])
    assert "configs/tasks/defaults/predator_capture_prey.yaml" in inventory
    assert "vendor/mapdn-source/COMMSTUDY_SOURCE_ORIGIN.json" in inventory


@pytest.mark.parametrize("radius", [1.0, None])
def test_native_pcp_observations_enforce_visibility_control_and_private_velocity(radius):
    from commstudy.tasks import resolve_task
    spec = default_plan()["pcp"]
    task = resolve_task("vmas_predator_capture_prey", {"predator_sensing_radius": radius})
    env = task.get_env_fun(1, True, 930001, "cpu")()
    try:
        for intervention in ("prey_position", "teammate_velocity"):
            states = [_pcp_setup(env, spec, intervention, state, 0.0, 930001)[2]
                      for state in range(2)]
            assert all(state["native_state_valid"] for state in states)
            comparison = observation_comparison(
                [s["recipient_observation"] for s in states],
                [s["donor_observation"] for s in states])
            assert comparison["recipient_exact_equal"] == (
                radius is not None or intervention == "teammate_velocity")
            assert not comparison["donor_exact_equal"]
            assert states[0]["exogenous_schedule"] == states[1]["exogenous_schedule"]
    finally:
        env.close()
