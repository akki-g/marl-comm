"""Allocation and temporal holdout failures must precede scientific collection."""

from copy import deepcopy

import pytest

from commstudy.experiments.next_phase import (
    MAPDN_CHECKPOINTS,
    choose_mapdn_budget,
    select_nonoverlapping_rows,
)


def calibration():
    return [
        {
            "method": "identity",
            "seed": seed,
            "integrity_passed": True,
            "validation": [
                {
                    "frames": frames,
                    "return": -10.0,
                    "voltage_magnitude": 0.001,
                    "violation_frequency": 0.1,
                    "q_effort": 0.2,
                    "solver_failures": 0,
                    "complete": True,
                }
                for frames in MAPDN_CHECKPOINTS
            ],
        }
        for seed in (40, 41)
    ]


def test_budget_is_method_blind_and_does_not_claim_convergence():
    data = calibration()
    decision = choose_mapdn_budget(data)
    assert decision["selected_frames"] == 16384
    assert decision["validation_stability_established"]
    assert not decision["convergence_established"]
    data[1]["method"] = "broadcast"
    with pytest.raises(ValueError, match="Identity"):
        choose_mapdn_budget(data)


def test_unstable_learning_selects_cap_without_declaring_adequacy():
    data = calibration()
    for run in data:
        for index, point in enumerate(run["validation"]):
            point["return"] = -10.0 + index * 3
    decision = choose_mapdn_budget(data)
    assert decision["selected_frames"] == 32768
    assert not decision["validation_stability_established"]
    assert decision["selection_reason"] == "declared_compute_cap"


@pytest.mark.parametrize(
    "fault",
    [
        "missing_seed",
        "duplicate_seed",
        "extra_checkpoint",
        "failed_integrity",
        "solver_failure",
        "nonfinite",
    ],
)
def test_invalid_baseline_evidence_cannot_allocate(fault):
    data = deepcopy(calibration())
    if fault == "missing_seed":
        data.pop()
    elif fault == "duplicate_seed":
        data[1]["seed"] = 40
    elif fault == "extra_checkpoint":
        data[0]["validation"][0]["frames"] = 1
    elif fault == "failed_integrity":
        data[0]["integrity_passed"] = False
    elif fault == "solver_failure":
        data[0]["validation"][1]["solver_failures"] = 1
    else:
        data[0]["validation"][1]["return"] = float("nan")
    with pytest.raises(ValueError):
        choose_mapdn_budget(data)


def test_excludes_history_and_terminal_row_not_just_start():
    rows = select_nonoverlapping_rows(
        range(2, 1000), 8, horizon=10, history=3, excluded=[(100, 150), (400, 500)]
    )
    for row in rows:
        footprint = set(range(row - 2, row + 11))
        assert not footprint.intersection(range(100, 150))
        assert not footprint.intersection(range(400, 500))
    assert all(b - 2 >= a + 11 for a, b in zip(rows[:-1], rows[1:], strict=True))


def test_rejects_episode_overlap_and_exhausted_holdout():
    with pytest.raises(ValueError, match="overlap"):
        select_nonoverlapping_rows(range(100), 8, horizon=50, history=1)
    with pytest.raises(ValueError, match="Insufficient"):
        select_nonoverlapping_rows(range(100), 8, horizon=5, history=1, excluded=[(0, 200)])
