"""Analysis fixtures contain no policies, environments, or optimization work."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def analysis():
    path = Path(__file__).resolve().parents[2] / "scripts" / "summarize_mapdn_confirmation.py"
    spec = importlib.util.spec_from_file_location("mapdn_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(analysis, *, row=0, observed=True):
    diagnostics = dict.fromkeys(analysis.PHYSICAL | analysis.ATTEMPTED, 0.0)
    diagnostics.update(
        valid=1.0,
        voltage_metrics_valid=float(observed),
        destroy=float(not observed),
        feasible_step=float(observed),
        action_solver_failure=float(not observed),
        voltage_bus_count=33,
        voltage_violating_bus_count=0,
        reward_row=row,
        observation_row=row + 1,
        average_voltage=1.0,
        q_loss=0.5,
        total_line_loss=0.2,
        reward=-1.0,
    )
    return {
        "step": row,
        "reward": -1.0,
        "terminated": not observed,
        "truncated": False,
        "actions": [[0.8], [-0.8], [0.0], [0.0], [0.0], [0.0]],
        "diagnostics": diagnostics,
    }


def raw_bank(analysis, effort=0.5):
    episodes = []
    for index in range(2):
        first = step(analysis, row=10 * index)
        first["diagnostics"]["q_loss"] = effort
        first["truncated"] = True
        episodes.append(
            {
                "episode_id": f"validation:{index}",
                "start_row": 10 * index,
                "seed": 200000 + index,
                "complete": True,
                "return": -1.0,
                "transition_count": 1,
                "transitions": [first],
            }
        )
    return {
        "split": "validation",
        "policy": "deterministic_actor",
        "episodes": episodes,
        "bank": [
            {
                key: value
                for key, value in episode.items()
                if key in {"episode_id", "start_row", "seed"}
            }
            for episode in episodes
        ],
    }


def test_terminal_failure_remains_attempted_but_stale_physics_excluded(analysis):
    first, failed = step(analysis), step(analysis, row=1, observed=False)
    failed["diagnostics"].update(total_line_loss=999.0, q_loss=999.0)
    result = analysis.transition_summary([first, failed], n_inverters=6)
    assert result["attempted_transition_denominator"] == 2
    assert result["physical_valid_transition_denominator"] == 1
    assert result["metrics"]["total_line_loss"]["mean"] == 0.2
    assert result["metrics"]["q_loss"]["mean"] == 0.5
    assert result["metrics"]["destroy"]["mean"] == 0.5
    assert result["metrics"]["destroy"]["eligible_denominator"] == 2
    assert result["voltage_frequency"]["bus_transition_denominator"] == 33
    assert result["actions"]["finite_action_scalar_denominator"] == 12
    assert result["actions"]["saturated_action_scalars"] == 4


def test_missing_and_nonfinite_values_report_separate_counts_and_null_means(analysis):
    first, second = step(analysis), step(analysis, row=1)
    del first["diagnostics"]["q_loss"]
    second["diagnostics"]["reward"] = float("nan")
    result = analysis.transition_summary([first, second], n_inverters=6)
    assert result["metrics"]["q_loss"]["missing_values"] == 1
    assert result["metrics"]["q_loss"]["mean"] is None
    assert result["metrics"]["q_loss"]["mean_over_finite_values"] == 0.5
    assert result["metrics"]["reward"]["nonfinite_values"] == 1
    assert result["metrics"]["reward"]["mean"] is None
    json.dumps(result, allow_nan=False)


def test_pairs_follow_exact_identity_not_episode_order_and_flag_missing(analysis):
    initial = analysis.bank_summary(raw_bank(analysis), n_inverters=6)
    raw_final = raw_bank(analysis, effort=0.25)
    raw_final["episodes"].reverse()
    final = analysis.bank_summary(raw_final, n_inverters=6)
    pairs = analysis.pair_banks(initial, final)
    assert pairs["pairing_valid"] and pairs["paired_episodes"] == 2
    assert all(
        pair["metric_mean_changes_final_minus_initial"]["q_loss"] == -0.25
        for pair in pairs["episodes"]
    )
    final["episodes"][0]["seed"] += 1
    pairs = analysis.pair_banks(initial, final)
    assert not pairs["pairing_valid"] and pairs["unpaired_episodes"] == 2


def test_tidy_health_counts_nonfinite_missing_and_missing_frames(analysis):
    csv = (
        "timestamp,frames,iteration,phase,group,metric,value,sample\n"
        "t,256,0,collection,agents,action_finite_fraction,1.0,\n"
        "t,512,1,collection,agents,action_finite_fraction,0.5,\n"
        "t,256,0,training,agents,loss_critic,nan,\n"
    )
    result = analysis.training_csv_summary(csv.encode(), [256, 512])
    assert result["nonfinite_value_csv_lines"] == [4]
    series = result["required_health_series"]
    assert series["collection/agents/action_finite_fraction"]["below_one_finite_fraction_rows"] == 1
    assert series["training/agents/loss_critic"]["missing_frames"] == [512]
    assert series["training/agents/grad_norm_loss_critic"]["missing_frames"] == [256, 512]
    assert not result["generic_action_saturated_fraction_used_for_physical_saturation"]


def test_every_read_input_is_hashed_and_mutation_or_escape_rejected(analysis, tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"value":1}')
    inputs = analysis.Inputs(tmp_path)
    assert inputs.json("input.json", "fixture") == {"value": 1}
    assert inputs.records["input.json"]["sha256"] == analysis.file_sha(path)
    with pytest.raises(ValueError, match="outside|escapes"):
        inputs.read(tmp_path.parent / "outside.json", "forbidden")
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="changed"):
        inputs.verify_unchanged()


def test_full_fixture_preserves_verdict_and_all_input_hashes(analysis, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    files = {
        "split_manifest.json": {
            "sha256": "manifest",
            "data": {
                "files": {},
                "interval_seconds": 180,
                "first_timestamp": "2012-01-01T00:03:00+00:00",
                "last_timestamp": "2015-01-01T00:00:00+00:00",
            },
        },
        "normalizer.json": {"sha256": "normalizer", "sample_count": 23040},
        "banks.json": {},
        "spec_seed30.json": {
            "experiment": {"on_policy_collected_frames_per_batch": 256, "max_n_frames": 8192}
        },
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    (root / "frozen_plan.md").write_text("Preserved fixture plan")
    packet = {
        "output": str(root),
        "allocation": [30],
        "protocol_id": "fixture",
        "feeder_dimensions": {"inverters": 6},
        "sources": {"commstudy_sha256": "source"},
        "files": {name: analysis.file_sha(root / name) for name in [*files, "frozen_plan.md"]},
    }
    packet["sha256"] = analysis._content_sha(packet)
    (root / "preparation.json").write_text(json.dumps(packet))
    recorded = {
        "verdict": "NOT_CONFIRMED",
        "scientific_learning_confirmed": False,
        "preparation_sha256": packet["sha256"],
        "training": [],
        "failure_stage": "training",
        "outcomes": [],
    }
    (root / "report.json").write_text(json.dumps(recorded))
    before = {path.name: analysis.file_sha(path) for path in root.iterdir()}
    result = analysis.summarize(root)
    assert result["original_protocol_verdict"] == "NOT_CONFIRMED"
    assert not result["verdict_recomputed_or_changed"]
    assert not result["evidence_consistency_issues"]
    assert result["seeds"]["30"]["missing_files"]
    assert result["measurement_context"]["source_row_interval_seconds"] == 180
    assert "zone-local" in result["measurement_context"]["actor_observations"]
    assert "not established" in result["measurement_context"]["timestamp_interpretation"]
    assert result["inputs"]["report.json"]["sha256"] == before["report.json"]
    assert before == {path.name: analysis.file_sha(path) for path in root.iterdir()}
    with pytest.raises(SystemExit):
        analysis.main(["--input-dir", str(root), "--out-file", str(root / "analysis.json")])
    output = tmp_path / "analysis.json"
    assert analysis.main(["--input-dir", str(root), "--out-file", str(output)]) == 0
    saved = copy.deepcopy(json.loads(output.read_text()))
    assert saved["original_protocol_verdict"] == "NOT_CONFIRMED"
    with pytest.raises(SystemExit):
        analysis.main(["--input-dir", str(root), "--out-file", str(output)])


def test_retained_real_smoke_bank_analysis_without_simulation(analysis):
    root = Path(__file__).resolve().parents[2]
    path = root / "results" / "mapdn_real_smoke_final_source_20260909" / "heldout_after.json"
    if not path.is_file():
        pytest.skip("Retained smoke fixture unavailable")
    raw = json.loads(path.read_text())
    result = analysis.bank_summary(raw, n_inverters=6)
    assert result["complete_episodes"] == 2
    assert result["pooled_domain"]["attempted_transition_denominator"] == 32
    assert result["pooled_domain"]["metrics"]["q_loss"]["mean"] == pytest.approx(
        raw["domain_means"]["q_loss"]
    )
    assert result["pooled_domain"]["actions"]["finite_action_scalar_denominator"] == 32 * 6
