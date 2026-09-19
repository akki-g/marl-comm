"""Synthetic artifact tests; no fixture is a scientific run or approval."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from commstudy.experiments.config import resolved_experiment_dict, scientific_config_sha256
from commstudy.experiments.protocols import content_sha256, load_protocol, protocol_spec
from commstudy.experiments.sweeps import RunPlan, write_manifest


ROOT = Path(__file__).resolve().parents[2]
LOADER = importlib.util.spec_from_file_location(
    "budget_analysis_script", ROOT / "scripts/analyze_pcp_budget.py"
)
analysis = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(analysis)
PROTOCOL = ROOT / "configs/protocols/pcp_visibility_budget_cpu_v1.yaml"
PLAN = ROOT / "docs/PCP_VISIBILITY_BUDGET_PLAN.md"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def episode(seed, *, model, arm, contact=False):
    metrics = dict.fromkeys(analysis.DOMAIN_KEYS, 0.0)
    metrics.update(
        episode_steps=100.0,
        complete=1.0,
        truncated=1.0,
        first_contact_step_or_horizon=25.0 if contact else 100.0,
        first_contact_observed=float(contact),
        any_contact=float(contact),
        contact_by_deadline=float(contact),
        contact_duration_steps=2.0 if contact else 0.0,
        contact_pair_steps=2.0 if contact else 0.0,
        contacting_predators_max=float(contact),
        contacting_predators_mean=0.02 if contact else 0.0,
        **{"return": 20.0 if contact else 0.0, "prey_visible_to_3_predators_mean": 1.0},
    )
    senders = (
        (0 if arm == "severed" else 2 if arm.startswith("suppress_") else 3)
        if model == "pcp_broadcast_k3"
        else 0
    )
    width = 32 if model in {"pcp_broadcast_k0", "pcp_broadcast_k3"} else 0
    return {
        "episode_id": f"pcp-heldout:{seed}",
        "seed": seed,
        "initial_physical_state_sha256": f"{seed:064x}",
        "exogenous_initial": {"exogenous_episode_ids": [0], "exogenous_steps": [0]},
        "metrics": metrics,
        "action_health": {
            "action_scalars": 600,
            "all_finite": True,
            "saturation_fraction": 0.0,
            "max_abs_action": 0.25,
            "max_abs_loc": 0.5,
            "min_scale": 0.5,
            "max_scale": 1.0,
        },
        "costs": {
            "sender_emissions": 100 * senders,
            "edge_deliveries": 200 * senders,
            "sender_payload_bits": 100 * senders * width * 32,
            "edge_delivery_payload_bits": 200 * senders * width * 32,
            "transitions": 100,
            "packet_scalars": width,
            "bits_per_scalar": 32,
            "cost_model": "one_float32_broadcast_packet_per_available_sender; two_receivers",
            "network_headers_counted": False,
            "measured_network_traffic": False,
        },
        "module_stats": {
            "messages_per_step": float(senders),
            "realized_sender_bits_per_step": float(senders * width * 32),
            "realized_bits_per_step": float(2 * senders * width * 32),
        },
    }


def make_evidence(root, model, visibility, seed):
    """Write clearly labelled synthetic evidence with real schema/hash bindings."""
    protocol = load_protocol(PROTOCOL)
    spec = protocol_spec(
        protocol, model=model, seed=seed, ablation="visibility", ablation_value=visibility
    )
    spec_hash = scientific_config_sha256(spec)
    protocol_hash = content_sha256(protocol)
    run_id = f"SYNTHETIC-{model}-{visibility}-{seed}"
    suite_id = "SYNTHETIC-pcp-budget-fixture"
    task_contract = {"fixture": "SYNTHETIC task contract, never a real run"}
    parameters = {"actor_total": 19332 if model == "pcp_comm_identity" else 27524}
    plan = RunPlan(
        run_id=run_id,
        suite_id=suite_id,
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
        command="SYNTHETIC: do not execute",
        resolved_spec=resolved_experiment_dict(spec),
        scientific_hash=spec_hash,
        protocol={
            "stage": "comparison",
            "model": model,
            "seed": seed,
            "ablation": "visibility",
            "ablation_value": visibility,
            "sha256": protocol_hash,
            "source_sha256": "a" * 64,
        },
        model_contract={"task_contract": task_contract, "parameter_counts": parameters},
    )
    run = plan.output_root / suite_id / run_id
    versions = {"python": protocol["runtime"]["python"], **protocol["runtime"]["packages"]}
    metadata = {
        "fixture": "SYNTHETIC ONLY",
        "status": "completed",
        "run_id": run_id,
        "suite_id": suite_id,
        "model": model,
        "seed": seed,
        "task": plan.task,
        "algorithm": "mappo",
        "ablation": "visibility",
        "ablation_value": visibility,
        "frames": 600000,
        "iterations": 100,
        "execution_purpose": "scientific",
        "source_sha256": "a" * 64,
        "scientific_config_sha256": spec_hash,
        "task_runtime_contract": task_contract,
        "parameters": parameters,
        "versions": versions,
        "protocol": {
            "protocol_sha256": protocol_hash,
            "source_sha256": "a" * 64,
            "stage": "comparison",
            "factors": {"model": model, "visibility": visibility},
            "approval_sha256": "b" * 64,
        },
        "runtime": {
            "sampling_device": "cpu",
            "train_device": "cpu",
            "buffer_device": "cpu",
            "torch_num_threads": 1,
        },
    }
    dump(run / "metadata.json", metadata)
    dump(run / "status.json", {"status": "completed"})
    (run / "resolved_config.yaml").write_text("# SYNTHETIC schema fixture\n")
    (run / "metrics.csv").write_text("SYNTHETIC schema fixture\n")
    (run / "checkpoints").mkdir()
    (run / "checkpoints/policy_state.pt").write_bytes(b"SYNTHETIC checkpoint; never load")
    evaluation = root / "evaluations" / run_id
    inventory = []
    for frame in range(120000, 600001, 120000):
        path = run / f"benchmarl/SYNTHETIC/checkpoints/checkpoint_{frame}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC header-validation dependency at {frame}".encode())
        inventory.append(
            {
                "path": str(path.resolve()),
                "sha256": analysis.sha256(path),
                "frames": frame,
                "iterations": frame // 6000,
            }
        )
    dump(
        evaluation / "retained_checkpoints.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "checkpoints": inventory,
        },
    )
    for name in ("action", "domain"):
        dump(evaluation / f"{name}_audit.json", {"fixture": f"SYNTHETIC {name} audit"})
    validation = {
        "run_id": run_id,
        "seed": seed,
        "stage": "comparison",
        "model": model,
        "ablation": "visibility",
        "ablation_value": visibility,
        "validation_passed": True,
        "issues": [],
        "artifacts": {
            name: analysis.sha256(run / name) for name in analysis.REQUIRED_RUN_ARTIFACTS
        },
        "action_audit_sha256": content_sha256({"fixture": "SYNTHETIC action audit"}),
        "domain_audit_sha256": content_sha256({"fixture": "SYNTHETIC domain audit"}),
        "performance": {
            "evaluation_frames": analysis.EVALUATION_FRAMES,
            "evaluation_returns": [float(index + seed) for index in range(51)],
        },
    }
    dump(evaluation / "training_validation.json", validation)
    report = {
        "run_id": run_id,
        "training_seed": seed,
        "model": model,
        "visibility": visibility,
        "protocol_sha256": protocol_hash,
        "plan_sha256": analysis.sha256(PLAN),
        "schema_version": 1,
        "evaluation_protocol": analysis.BUDGET_EVALUATION_PROTOCOL,
        "group": "adversary",
        "exploration": "DETERMINISTIC",
        "deadline": 50,
        "episode_seeds": analysis.EPISODE_SEEDS,
        "shuffle_definition": "frozen_live_packets_from_cyclic_next_episode_at_same_step",
        "shuffle_distribution_shift": (
            "donor live trajectories substituted into receiver closed-loop trajectories"
        ),
        "analysis_provenance": {
            "source_run": str(run.resolve()),
            "source_sha256": "a" * 64,
            "checkpoint_sha256": validation["artifacts"]["checkpoints/policy_state.pt"],
            "resolved_config_sha256": validation["artifacts"]["resolved_config.yaml"],
            "scientific_config_sha256": spec_hash,
            "versions": versions,
            "task_runtime_contract": task_contract,
            "allow_runtime_mismatch": False,
            "compatibility_mismatches": {},
            "analysis_overrides": {
                "save_folder": str(root / "SYNTHETIC-scratch"),
                "loggers": [],
                "create_json": False,
                "checkpoint_interval": 0,
                "checkpoint_at_end": False,
                "restore_file": None,
                "restore_map_location": None,
                "render": False,
            },
        },
        "training_validation_sha256": analysis.sha256(evaluation / "training_validation.json"),
        "retained_checkpoints_sha256": analysis.sha256(evaluation / "retained_checkpoints.json"),
        "live_packet_bank_sha256": None,
        "interventions": {},
    }
    for arm in analysis.INTERVENTIONS if model == "pcp_broadcast_k3" else ("live", "severed"):
        successes = {
            "pcp_comm_identity": 64,
            "pcp_local_capacity": 128,
            "pcp_broadcast_k0": 64 if visibility == "radius1" else 224,
            "pcp_broadcast_k3": 224 if arm == "live" else 64,
        }[model]
        report["interventions"][arm] = [
            episode(ep_seed, model=model, arm=arm, contact=index < successes)
            for index, ep_seed in enumerate(analysis.EPISODE_SEEDS)
        ]
    if model == "pcp_broadcast_k3":
        # Expanded storage keeps fixtures small; the full tensor's shape/hash is checked.
        bank = torch.zeros(1, dtype=torch.float32).expand(256, 100, 1, 3, 32)
        path = evaluation / "live_packet_bank.pt"
        torch.save({"packets": bank, "episode_seeds": analysis.EPISODE_SEEDS}, path)
        report["packet_file_sha256"] = analysis.sha256(path)
        report["live_packet_bank_sha256"] = analysis.tensor_sha256(bank)
        donor_hash = analysis.tensor_sha256(bank[0])
        for index, row in enumerate(report["interventions"]["shuffle"]):
            row.update(
                donor_episode_seed=analysis.EPISODE_SEEDS[(index + 1) % 256],
                donor_packet_sha256=donor_hash,
            )
    dump(evaluation / "heldout_evaluation.json", report)
    dump(evaluation / "status.json", {"status": "completed"})
    return SimpleNamespace(plan=plan, run=run, evaluation=evaluation, report=report)


@pytest.fixture(scope="module")
def complete_grid(tmp_path_factory):
    root = tmp_path_factory.mktemp("SYNTHETIC-budget-comparison")
    fixtures = [
        make_evidence(root, model, visibility, seed)
        for model, visibility, seed in itertools.product(
            analysis.MODELS, analysis.VISIBILITIES, analysis.TRAINING_SEEDS
        )
    ]
    manifest = root / "manifest.csv"
    write_manifest(manifest, [fixture.plan for fixture in fixtures])
    report = analysis.analyze(
        manifest, root / "evaluations", protocol_path=PROTOCOL, plan_path=PLAN
    )
    return SimpleNamespace(root=root, fixtures=fixtures, manifest=manifest, report=report)


def test_full_synthetic_grid_aggregates_policies_before_seed_bootstrap(complete_grid):
    report = complete_grid.report
    assert report["checks_passed"], [(r["run_id"], r["issues"]) for r in report["rows"]]
    assert report["valid_policies"] == report["expected_policies"] == 40
    assert report["approval_decision"] == "not_issued"
    primary = next(c for c in report["contrasts"] if c["contrast"] == report["primary_contrast"])
    assert primary["interval"]["mean"] == 0.375
    assert primary["interval"]["training_seeds"] == 5
    assert primary["interval"]["bootstrap_samples"] == 10000
    assert [v["value"] for v in primary["per_seed"]] == [0.375] * 5
    assert [r["smallest_tested_budget"] for r in report["budget_frontiers"]] == [3, 0]
    assert all(row["arms"]["live"]["episodes"] == 256 for row in report["rows"])
    assert all(row["arms"]["live"]["action_scalars"] == 153600 for row in report["rows"])
    assert report["training_source_sha256"] == "a" * 64
    assert report["analysis_source_sha256"] != report["training_source_sha256"]
    assert len(report["artifact_hashes"]) >= 40 * 12


def test_complete_synthetic_outputs_preserve_axis_and_provenance(complete_grid):
    out = complete_grid.root / "SYNTHETIC-outputs"
    analysis.write_outputs(complete_grid.report, out)
    saved = json.loads((out / "analysis_report.json").read_text())
    assert all("episode_records" not in row for row in saved["rows"])
    assert len((out / "training_curves.csv").read_text().splitlines()) == 40 * 51 + 1
    assert len((out / "per_seed.csv").read_text().splitlines()) == 120 + 1
    assert len((out / "episode_metrics.csv").read_text().splitlines()) == 120 * 256 + 1
    for suffix in ("png", "svg", "pdf"):
        assert len(list(out.glob(f"*.{suffix}"))) == 3
    provenance = json.loads((out / "provenance.json").read_text())
    assert all(
        analysis.sha256(out / path) == digest
        for path, digest in provenance["output_artifact_hashes"].items()
    )
    assert provenance["no_training_or_policy_rollout"] is True
    with pytest.raises(ValueError, match="overwrite"):
        analysis.write_outputs(complete_grid.report, out)


def test_missing_seed_never_gets_reduced_seed_interval_or_frontier(complete_grid):
    records = {
        (r["model"], r["visibility"], r["seed"]): dict(r) for r in complete_grid.report["rows"]
    }
    records[("pcp_broadcast_k3", "radius1", 24)]["validation_passed"] = False
    contrasts, frontiers = analysis.summarize(records)
    affected = next(
        c for c in contrasts if c["contrast"] == complete_grid.report["primary_contrast"]
    )
    assert affected["missing_or_invalid_seeds"] == [24]
    assert len(affected["per_seed"]) == 4 and affected["interval"] is None
    assert frontiers[0]["smallest_tested_budget"] == "unavailable"
    assert frontiers[1]["smallest_tested_budget"] == 0


def test_frontier_uses_lower_bound_not_mean_or_untested_budget(complete_grid):
    records = {
        (r["model"], r["visibility"], r["seed"]): {**r, "arms": copy.deepcopy(r["arms"])}
        for r in complete_grid.report["rows"]
    }
    for index, seed in enumerate(analysis.TRAINING_SEEDS):
        records[("pcp_broadcast_k3", "radius1", seed)]["arms"]["live"]["contact_by_deadline"] = [
            0.2,
            1,
            1,
            1,
            1,
        ][index]
    _, frontiers = analysis.summarize(records)
    assert frontiers[0]["budgets"][1]["interval"]["mean"] > 0.8
    assert frontiers[0]["budgets"][1]["interval"]["ci95_low"] < 0.8
    assert frontiers[0]["smallest_tested_budget"] == "not reached"


@pytest.mark.parametrize("failure", ["unplanned_row", "mixed_source"])
def test_global_cohort_integrity_failure_suppresses_all_intervals(
    complete_grid,
    monkeypatch,
    failure,
):
    plans = [fixture.plan for fixture in complete_grid.fixtures]
    # Extra planned rows are a cohort error even though the original 40 rows
    # remain individually valid. Real files are unchanged by this synthetic test.
    from dataclasses import replace

    if failure == "unplanned_row":
        plans.append(replace(plans[0], model="UNPLANNED-MODEL", run_id="UNPLANNED-SYNTHETIC"))
    else:
        plans[0] = replace(plans[0], protocol={**plans[0].protocol, "source_sha256": "c" * 64})
    monkeypatch.setattr(analysis, "read_manifest", lambda _path: plans)
    report = analysis.analyze(
        complete_grid.manifest,
        complete_grid.root / "evaluations",
        protocol_path=PROTOCOL,
        plan_path=PLAN,
    )
    assert report["valid_policies"] == (40 if failure == "unplanned_row" else 39)
    assert not report["checks_passed"]
    if failure == "mixed_source":
        assert any("common training source" in issue for issue in report["issues"])
    assert all(c["interval"] is None for c in report["contrasts"])
    assert all(f["smallest_tested_budget"] == "unavailable" for f in report["budget_frontiers"])
    assert all(c["means"] == c["intervals"] == {} for c in report["conditions"])


@pytest.mark.parametrize(
    "change,pattern",
    [
        (lambda m: m.update(first_contact_step_or_horizon=25.5), "Fractional"),
        (
            lambda m: m.update(
                contact_duration_steps=1, contact_pair_steps=300, **{"return": 3000}
            ),
            "Impossible contact",
        ),
        (lambda m: m.update(first_contact_step_or_horizon=100, contact_by_deadline=0), "precedes"),
        (lambda m: m.update(contacting_predators_mean=2), "predator count"),
    ],
)
def test_impossible_transition_counts_are_rejected(change, pattern):
    row = episode(200000, model="pcp_comm_identity", arm="live", contact=True)
    change(row["metrics"])
    with pytest.raises(ValueError, match=pattern):
        analysis.validate_episode(row, seed=200000, model="pcp_comm_identity", arm="live")


@pytest.mark.parametrize(
    "duration,simultaneous,pairs,maximum",
    [
        (5, 0, 5, 1),
        (5, 2, 7, 2),
        (5, 2, 9, 3),
        (1, 1, 3, 3),
    ],
)
def test_valid_one_two_three_predator_contact_histories(duration, simultaneous, pairs, maximum):
    row = episode(200000, model="pcp_comm_identity", arm="live", contact=True)
    row["metrics"].update(
        contact_duration_steps=duration,
        simultaneous_contact_steps=simultaneous,
        contact_pair_steps=pairs,
        contacting_predators_max=maximum,
        contacting_predators_mean=pairs / 100,
        any_simultaneous_contact=float(simultaneous > 0),
        **{"return": 10 * pairs},
    )
    analysis.validate_episode(row, seed=200000, model="pcp_comm_identity", arm="live")
    row["metrics"]["contact_pair_steps"] += 10
    with pytest.raises(ValueError, match="Impossible contact"):
        analysis.validate_episode(row, seed=200000, model="pcp_comm_identity", arm="live")


@pytest.mark.parametrize(
    "change,pattern",
    [
        (lambda row: row["metrics"].update(truncated=0), "time-limit"),
        (lambda row: row["metrics"].update(episode_steps=99), "time-limit"),
        (lambda row: row["metrics"].update(contact_by_deadline=1), "Deadline"),
        (lambda row: row["metrics"].update(reward_accounting_max_abs_error=1), "reward"),
        (lambda row: row["metrics"].update(predator_boundary_fraction=float("nan")), "non-finite"),
        (lambda row: row["action_health"].update(action_scalars=800), "scalar count"),
        (lambda row: row["action_health"].update(min_scale=0), "scale"),
        (lambda row: row["costs"].update(sender_emissions=1), "payload"),
        (lambda row: row["exogenous_initial"].update(exogenous_steps=[1]), "time zero"),
    ],
)
def test_invalid_episode_rejected(change, pattern):
    row = episode(200000, model="pcp_comm_identity", arm="live")
    change(row)
    with pytest.raises(ValueError, match=pattern):
        analysis.validate_episode(row, seed=200000, model="pcp_comm_identity", arm="live")


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("missing_episode", "exactly 256"),
        ("duplicate_episode", "ordering"),
        ("wrong_group", "group"),
        ("deadline", "deadline"),
        ("null_health", "exact null"),
        ("runtime_acknowledgement", "runtime binding"),
        ("validation_hash", "validation digest"),
        ("missing_native", "No such file"),
        ("changed_native", "checkpoint changed"),
        ("changed_metric_artifact", "artifact changed"),
        ("failed_validation", "did not pass"),
    ],
)
def test_invalid_saved_dependencies_fail_closed(tmp_path, mutation, pattern):
    item = make_evidence(tmp_path, "pcp_local_capacity", "radius1", 20)
    report = item.report
    if mutation == "missing_episode":
        report["interventions"]["live"].pop()
    elif mutation == "duplicate_episode":
        report["interventions"]["live"][1] = report["interventions"]["live"][0]
    elif mutation == "wrong_group":
        report["group"] = "agent"
    elif mutation == "deadline":
        report["deadline"] = 51
    elif mutation == "null_health":
        report["interventions"]["severed"][0]["action_health"]["max_abs_loc"] = 0.6
    elif mutation == "runtime_acknowledgement":
        report["analysis_provenance"]["allow_runtime_mismatch"] = True
    elif mutation == "validation_hash":
        report["training_validation_sha256"] = "c" * 64
    elif mutation in {"missing_native", "changed_native"}:
        path = next(item.run.rglob("checkpoint_480000.pt"))
        path.unlink() if mutation == "missing_native" else path.write_bytes(b"changed")
    elif mutation == "changed_metric_artifact":
        (item.run / "metrics.csv").write_text("changed artifact")
    elif mutation == "failed_validation":
        path = item.evaluation / "training_validation.json"
        validation = json.loads(path.read_text())
        validation["validation_passed"] = False
        dump(path, validation)
        report["training_validation_sha256"] = analysis.sha256(path)
    dump(item.evaluation / "heldout_evaluation.json", report)
    with pytest.raises((ValueError, FileNotFoundError), match=pattern):
        analysis.validate_row(
            item.plan,
            item.evaluation,
            protocol=load_protocol(PROTOCOL),
            protocol_hash=content_sha256(load_protocol(PROTOCOL)),
            plan_hash=analysis.sha256(PLAN),
            hashes={},
        )


def test_packet_shuffle_requires_declared_cyclic_donor(tmp_path):
    item = make_evidence(tmp_path, "pcp_broadcast_k3", "radius1", 20)
    item.report["interventions"]["shuffle"][0]["donor_episode_seed"] = 200255
    dump(item.evaluation / "heldout_evaluation.json", item.report)
    with pytest.raises(ValueError, match="donor identity"):
        analysis.validate_row(
            item.plan,
            item.evaluation,
            protocol=load_protocol(PROTOCOL),
            protocol_hash=content_sha256(load_protocol(PROTOCOL)),
            plan_hash=analysis.sha256(PLAN),
            hashes={},
        )


def test_corrupt_packet_file_preserves_invalid_row_in_complete_report(tmp_path):
    item = make_evidence(tmp_path, "pcp_broadcast_k3", "radius1", 20)
    path = item.evaluation / "live_packet_bank.pt"
    path.write_bytes(b"SYNTHETIC deliberately corrupt pickle fixture")
    item.report["packet_file_sha256"] = analysis.sha256(path)
    dump(item.evaluation / "heldout_evaluation.json", item.report)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [item.plan])
    report = analysis.analyze(
        manifest,
        tmp_path / "evaluations",
        protocol_path=PROTOCOL,
        plan_path=PLAN,
    )
    assert len(report["rows"]) == 40 and not report["checks_passed"]
    row = next(row for row in report["rows"] if row["run_id"] == item.plan.run_id)
    assert row["status"] == "invalid"
    assert "UnpicklingError" in row["issues"][0]


def test_incomplete_grid_keeps_all_forty_expected_rows_and_failed_status(tmp_path):
    item = make_evidence(tmp_path, "pcp_comm_identity", "radius1", 20)
    dump(item.run / "status.json", {"status": "failed"})
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [item.plan, item.plan])
    report = analysis.analyze(
        manifest, tmp_path / "evaluations", protocol_path=PROTOCOL, plan_path=PLAN
    )
    assert len(report["rows"]) == 40 and not report["checks_passed"]
    assert any("40 unique" in issue for issue in report["issues"])
    assert any("duplicate run IDs" in issue for issue in report["issues"])
    write_manifest(manifest, [item.plan])
    report = analysis.analyze(
        manifest, tmp_path / "evaluations", protocol_path=PROTOCOL, plan_path=PLAN
    )
    row = next(row for row in report["rows"] if row["run_id"] == item.plan.run_id)
    assert row["status"] == "failed"
    assert report["valid_policies"] == 0
    assert all(contrast["interval"] is None for contrast in report["contrasts"])


def test_seed_bootstrap_does_not_treat_episodes_as_independent():
    result = analysis.paired_seed_interval([0.0, 0.25, 0.5, 0.75, 1.0])
    rng = np.random.default_rng(73191)
    samples = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])[rng.integers(0, 5, (10000, 5))].mean(1)
    assert result["mean"] == 0.5
    assert (result["ci95_low"], result["ci95_high"]) == tuple(np.quantile(samples, [0.025, 0.975]))


@pytest.mark.parametrize("model", ["pcp_local_capacity", "pcp_broadcast_k3"])
def test_real_evaluator_episode_matches_standalone_analysis_schema(tmp_path, model):
    from commstudy.analysis.pcp_budget import _episode
    from commstudy.experiments import build_experiment

    spec = protocol_spec(
        load_protocol(PROTOCOL),
        model=model,
        seed=20,
        ablation="visibility",
        ablation_value="radius1",
    )
    spec.experiment.update(
        save_folder=str(tmp_path / "SYNTHETIC-untrained-schema-check"),
        checkpoint_interval=0,
        checkpoint_at_end=False,
        create_json=False,
        evaluation=False,
        loggers=[],
    )
    experiment = build_experiment(spec)
    try:
        row, _ = _episode(experiment, seed=200000, intervention="live", donor=None, deadline=50)
        analysis.validate_episode(row, seed=200000, model=model, arm="live")
    finally:
        experiment.close()
