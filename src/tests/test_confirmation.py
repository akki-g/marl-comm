"""Synthetic artifact checks, with no approval issuance or training work."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from commstudy.analysis.confirmation import (
    AUDIT_BASE_SEED,
    AUDIT_EPISODES,
    COLLECTION_HEALTH_METRICS,
    ONLINE_DOMAIN_METRICS,
    TRAINING_HEALTH_METRICS,
    _validate_run,
    confirm_pcp_suite,
    expected_evaluation_frames,
    file_sha256,
)
from commstudy.analysis.pcp_domain import summarize_domain_episodes
from commstudy.experiments.config import resolved_experiment_dict, scientific_config_sha256
from commstudy.experiments.metrics import METRIC_COLUMNS
from commstudy.experiments.protocols import content_sha256, load_protocol, protocol_spec
from commstudy.experiments.provenance import source_fingerprint


def _episode_metrics(contact_steps=0):
    observed = float(contact_steps > 0)
    return {
        "episode_steps": 100.0,
        "complete": 1.0,
        "terminated": 0.0,
        "truncated": 1.0,
        "return": 10.0 * contact_steps,
        "any_contact": observed,
        "first_contact_observed": observed,
        "first_contact_step_or_horizon": 1.0 if observed else 100.0,
        "contact_duration_steps": float(contact_steps),
        "contact_pair_steps": float(contact_steps),
        "simultaneous_contact_steps": 0.0,
        "any_simultaneous_contact": 0.0,
        "contacting_predators_mean": contact_steps / 100.0,
        "contacting_predators_max": observed,
        "nearest_predator_prey_distance_mean": 0.5,
        "nearest_predator_prey_distance_min": 0.0,
        "mean_predator_nearest_prey_distance": 0.5,
        "predator_boundary_fraction": 0.0,
        "prey_boundary_fraction": 0.0,
        "predator_obstacle_collision_pair_steps": 0.0,
        "predator_teammate_collision_pair_steps": 0.0,
        "sender_visible_receiver_blind_pairs_mean": 0.0,
        "sender_visible_receiver_blind_prey_triples_mean": 0.0,
        "reward_accounting_max_abs_error": 0.0,
        **{f"prey_visible_to_{k}_predators_mean": float(k == 3) for k in range(4)},
    }


def _write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def confirmation_case(tmp_path, config_root):
    root = tmp_path / "confirmation_repository"
    package = root / "src" / "commstudy"
    package.mkdir(parents=True)
    (package / "fixture.py").write_text("FIXTURE = 1\n")
    protocol = load_protocol(config_root / "protocols" / "pcp_corrected_v1.yaml")
    path = root / "protocol.yaml"
    OmegaConf.save(OmegaConf.create(protocol), path)
    source_hash = source_fingerprint(root)
    protocol_hash = content_sha256(protocol)
    suite = root / "runs" / "synthetic_confirmation"
    suite.mkdir(parents=True)
    actions, domains, rows_by_run = {}, {}, {}
    for seed in protocol["selection"]["confirmation_seeds"]:
        run_id = f"identity_seed_{seed}"
        run = suite / run_id
        run.mkdir()
        (run / "checkpoints").mkdir()
        spec = protocol_spec(protocol, model="pcp_comm_identity", seed=seed)
        torch.save(
            {
                "schema_version": 1,
                "run_id": run_id,
                "suite_id": suite.name,
                "algorithm": spec.algorithm,
                "task": spec.task,
                "model": spec.model,
                "seed": seed,
                "frames": 600000,
                "group_policies": {
                    "adversary": {"weight": torch.ones(1)},
                    "agent": {"weight": torch.ones(1)},
                },
            },
            run / "checkpoints" / "policy_state.pt",
        )
        scientific_hash = scientific_config_sha256(spec)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "commstudy": resolved_experiment_dict(spec),
                    "scientific_config_sha256": scientific_hash,
                }
            ),
            run / "resolved_config.yaml",
        )
        metadata = {
            "run_id": run_id,
            "suite_id": suite.name,
            "status": "completed",
            "seed": seed,
            "task": spec.task,
            "algorithm": "mappo",
            "model": "pcp_comm_identity",
            "execution_purpose": "scientific",
            "ablation": "main",
            "ablation_value": None,
            "frames": 600000,
            "iterations": 100,
            "source_sha256": source_hash,
            "scientific_config_sha256": scientific_hash,
            "protocol": {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_hash,
                "source_sha256": source_hash,
                "stage": "confirmation",
                "factors": {"model": "pcp_comm_identity", "main": None},
                "approval_sha256": "a" * 64,
            },
            "runtime": {
                "torch_num_threads": protocol["runtime"]["threads"],
                **{
                    key: protocol["runtime"]["device"]
                    for key in ("sampling_device", "train_device", "buffer_device")
                },
            },
            "versions": {
                "python": protocol["runtime"]["python"],
                **protocol["runtime"]["packages"],
            },
            "task_runtime_contract": {
                **protocol["contracts"],
                "critic_state": {"protocol": protocol["contracts"]["critic_state_protocol"]},
            },
        }
        (run / "metadata.json").write_text(json.dumps(metadata))
        (run / "status.json").write_text(json.dumps({"status": "completed"}))
        provenance = {
            "source_run": str(run),
            "resolved_config_sha256": file_sha256(run / "resolved_config.yaml"),
            "checkpoint_sha256": file_sha256(run / "checkpoints" / "policy_state.pt"),
            "scientific_config_sha256": scientific_hash,
            "source_sha256": source_hash,
            "versions": metadata["versions"],
            "task_runtime_contract": metadata["task_runtime_contract"],
            "analysis_overrides": {},
            "allow_runtime_mismatch": False,
            "compatibility_mismatches": {},
        }
        action = {
            "run_id": run_id,
            "measured_group": "adversary",
            "policy_audited": True,
            "analysis_provenance": copy.deepcopy(provenance),
        }
        for mode in ("deterministic", "random"):
            values = {
                "episodes": AUDIT_EPISODES,
                "steps": 100,
                "complete_episodes": AUDIT_EPISODES,
                "group": "adversary",
                "episode_transitions": [100] * AUDIT_EPISODES,
                "episode_ids": [
                    f"seed:{AUDIT_BASE_SEED}/episode:{i}" for i in range(AUDIT_EPISODES)
                ],
                "action_scalars": 32 * 100 * 3 * 2,
                "finite_action_scalars": 32 * 100 * 3 * 2,
                "finite_action_fraction": 1.0,
                "finite_location_fraction": 1.0,
                "finite_scale_fraction": 1.0,
                "saturation_denominator": "finite_action_scalars",
                "saturation_threshold": 0.999,
                "saturated_action_fraction": 0.0,
                "mean_abs_action": 0.1,
                "max_abs_action": 0.5,
                "mean_abs_location": 0.1,
                "max_abs_location": 0.5,
                "mean_scale": 1.0,
                "max_scale": 1.1,
                **{
                    name: dict.fromkeys(("p05", "p50", "p95", "p99"), 0.1)
                    for name in (
                        "action_abs_quantiles",
                        "location_abs_quantiles",
                        "scale_quantiles",
                    )
                },
            }
            action.update({f"{mode}_{key}": value for key, value in values.items()})
        actions[run_id] = action
        domain = {
            "schema_version": 1,
            "domain_protocol": "pcp_transition_metrics_v1",
            "group": "adversary",
            "analysis_provenance": copy.deepcopy(provenance),
            "modes": {},
        }
        for mode in ("DETERMINISTIC", "RANDOM"):
            episodes = [
                {
                    "episode_id": f"seed:{AUDIT_BASE_SEED}/episode:{i}",
                    "seed": AUDIT_BASE_SEED + i,
                    "metrics": _episode_metrics(i % 2),
                }
                for i in range(AUDIT_EPISODES)
            ]
            domain["modes"][mode] = {"episodes": episodes, **summarize_domain_episodes(episodes)}
        domains[run_id] = domain
        rows = []

        def add(frame, phase, group, metric, value, sample="", rows=rows):
            rows.append(
                dict(
                    timestamp="fixture",
                    frames=frame,
                    iteration=frame // 6000 - 1,
                    phase=phase,
                    group=group,
                    metric=metric,
                    value=value,
                    sample=sample,
                )
            )

        for frame in range(6000, 600001, 6000):
            for group, agents in [("adversary", 3), ("agent", 1)]:
                for metric in TRAINING_HEALTH_METRICS:
                    value = (
                        60000 * agents
                        if metric.endswith("_scalar_count")
                        else 60000
                        if metric == "optimization_transition_count"
                        else -0.2
                        if metric == "explained_variance"
                        else 1.0
                    )
                    add(frame, "training", group, metric, value)
                for metric in COLLECTION_HEALTH_METRICS:
                    value = (
                        6000
                        if metric in ("health_transition_count", "replay_transition_count")
                        else 6000 * agents * 2
                        if metric.endswith("_scalar_count")
                        else 0.0
                        if metric in ("action_saturated_fraction", "replay_log_prob_max_abs_error")
                        else 1.0
                    )
                    add(frame, "collection", group, metric, value)
            for metric, value in {
                "transition_count": 6000,
                "contact_pairs_mean": 0.1,
                "predator_reward_mean": 1.0,
                "reward_accounting_max_abs_error": 0.0,
            }.items():
                add(frame, "collection_domain", "adversary", metric, value)
        for index, frame in enumerate(expected_evaluation_frames(protocol)):
            # All episodes have an integral contact duration and exact10x reward.
            metrics = _episode_metrics(index)
            for group in ("", "adversary"):
                add(frame, "evaluation", group, "return_mean", metrics["return"])
            add(frame, "evaluation", "", "episode_length_mean", 100)
            add(frame, "evaluation", "adversary", "health_transition_count", 12800)
            for episode in range(128):
                add(frame, "evaluation", "", "return_episode", metrics["return"], str(episode))
                for metric in ONLINE_DOMAIN_METRICS:
                    add(
                        frame,
                        "evaluation_domain_episode",
                        "adversary",
                        metric,
                        metrics[metric],
                        str(episode),
                    )
            for metric in ONLINE_DOMAIN_METRICS:
                add(frame, "evaluation_domain", "adversary", metric, metrics[metric])
            for metric, value in {
                "episode_count": 128,
                "complete_episodes": 128,
                "transition_count": 12800,
            }.items():
                add(frame, "evaluation_domain", "adversary", metric, value)
        rows_by_run[run_id] = rows
        _write_rows(run / "metrics.csv", rows)
    case = SimpleNamespace(
        root=root,
        suite=suite,
        protocol=protocol,
        path=path,
        actions=actions,
        domains=domains,
        rows=rows_by_run,
    )
    case.confirm = lambda: confirm_pcp_suite(
        suite, protocol_path=path, repo_root=root, action_audits=actions, domain_audits=domains
    )
    return case


def _checks(report):
    return {issue["check"] for issue in report["issues"]} | {
        issue["check"] for run in report["runs"] for issue in run["issues"]
    }


def _comparison_case(case):
    """Relabel one complete synthetic artifact row under a declared comparison factor."""
    model, factor, value = 'pcp_comm_attention', 'visibility', 'global'
    condition = {'stage': 'comparison', 'model': model,
                 'ablation': factor, 'ablation_value': value}
    case.protocol['selection']['comparison_seeds'] = [10, 11]
    case.protocol['factors'][factor] = {
        'models': [model], 'values': {value: {
            model: {'task_config.params.predator_sensing_radius': None},
        }},
    }
    case.protocol['launch_scope'] = {'comparison': [
        {key: condition[key] for key in ('model', 'ablation', 'ablation_value')},
    ]}
    protocol_hash = content_sha256(case.protocol)
    run = case.suite / sorted(case.rows)[0]
    metadata = json.loads((run / 'metadata.json').read_text())
    seed = metadata['seed']
    spec = protocol_spec(case.protocol, model=model, seed=seed,
                         ablation=factor, ablation_value=value)
    scientific_hash = scientific_config_sha256(spec)
    OmegaConf.save(OmegaConf.create({
        'commstudy': resolved_experiment_dict(spec), 'scientific_config_sha256': scientific_hash,
    }), run / 'resolved_config.yaml')
    checkpoint = run / 'checkpoints' / 'policy_state.pt'
    policy = torch.load(checkpoint, weights_only=True)
    policy['model'] = model
    torch.save(policy, checkpoint)
    metadata.update({key: condition[key] for key in ('model', 'ablation', 'ablation_value')})
    metadata['scientific_config_sha256'] = scientific_hash
    metadata['protocol'].update({
        'stage': 'comparison', 'protocol_sha256': protocol_hash,
        'factors': {'model': model, factor: value},
    })
    (run / 'metadata.json').write_text(json.dumps(metadata))
    for audit in (case.actions[run.name], case.domains[run.name]):
        audit['analysis_provenance'].update({
            'resolved_config_sha256': file_sha256(run / 'resolved_config.yaml'),
            'checkpoint_sha256': file_sha256(checkpoint),
            'scientific_config_sha256': scientific_hash,
        })
    return run, metadata, condition


def test_comparison_condition_runs_full_artifact_gate_and_retains_exact_labels(confirmation_case):
    case = confirmation_case
    run, metadata, condition = _comparison_case(case)
    result = _validate_run(
        run, metadata, case.protocol, content_sha256(case.protocol), source_fingerprint(case.root),
        case.actions[run.name], case.domains[run.name], condition=condition,
    )
    assert result['validation_passed'], result['issues']
    assert all(result[key] == value for key, value in condition.items())
    assert result['performance']['final_window_mean_return'] == 475


@pytest.mark.parametrize('corruption,check', [
    ('implicit_confirmation', 'declared_condition'),
    ('undeclared_condition', 'declared_condition'),
    ('undeclared_seed', 'declared_condition'),
    ('nested_factor', 'protocol_factors'),
    ('metadata_condition', 'run_identity'),
    ('incomplete_horizon', 'run_identity'),
])
def test_comparison_reuses_strict_scope_labels_and_horizon_gate(
    confirmation_case, corruption, check,
):
    case = confirmation_case
    run, metadata, condition = _comparison_case(case)
    if corruption == 'implicit_confirmation':
        condition = None
    elif corruption == 'undeclared_condition':
        condition['ablation_value'] = 'radius1'
    elif corruption == 'undeclared_seed':
        metadata['seed'] = 999
    elif corruption == 'nested_factor':
        metadata['protocol']['factors']['visibility'] = 'radius1'
    elif corruption == 'metadata_condition':
        metadata['ablation_value'] = 'radius1'
    else:
        metadata['frames'] = 594000
    result = _validate_run(
        run, metadata, case.protocol, content_sha256(case.protocol), source_fingerprint(case.root),
        case.actions[run.name], case.domains[run.name], condition=condition,
    )
    assert not result['validation_passed']
    assert check in {item['check'] for item in result['issues']}


def test_complete_confirmation_requires_review_and_reports_prespecified_windows(confirmation_case):
    case = confirmation_case
    report = case.confirm()
    assert report["checks_passed"], report
    assert report["review_status"] == "required"
    assert report["approval_decision"] == "not_issued"
    assert report["expected_seeds"] == [10, 11]
    assert report["expected_evaluation_frames"] == [6000, *range(12000, 600001, 12000)]
    for run in report["runs"]:
        performance = run["performance"]
        assert performance["final_window_points"] == 6
        assert performance["final_window_mean_return"] == 475
        assert performance["first_logged_evaluation_frames"] == 6000
        assert performance["final_minus_first_logged_evaluation"] == 475
        assert performance["early_window_mean"] == 25
        assert performance["late_window_mean"] == 475
        assert performance["early_late_delta"] == 450
        assert performance["auc_frame_interval"] == [6000, 600000]
        assert performance["auc"] == 149970000
        assert performance["normalized_auc"] == pytest.approx(149970000 / 594000)
        domain = run["domain_learning_summary"]["contact_duration_steps"]
        assert domain["early_window_mean"] == 2.5
        assert domain["late_window_mean"] == 47.5
        assert domain["early_late_delta"] == 45
        assert len(run["artifacts"]["metrics.csv"]) == 64
    assert not list(case.root.rglob("*approval*.json"))


@pytest.mark.parametrize(
    "corruption,check",
    [
        ("failed_status", "completed_run"),
        ("short_horizon", "run_identity"),
        ("stale_source", "current_source"),
        ("wrong_protocol", "protocol_binding"),
        ("wrong_seed", "exact_seed_coverage"),
        ("runtime_change", "saved_runtime"),
    ],
)
def test_incomplete_stale_or_mismatched_runs_are_not_confirmation(
    confirmation_case, corruption, check
):
    case = confirmation_case
    run = case.suite / next(iter(case.actions))
    metadata = json.loads((run / "metadata.json").read_text())
    if corruption == "failed_status":
        (run / "status.json").write_text(json.dumps({"status": "failed"}))
    elif corruption == "short_horizon":
        metadata["frames"] = 594000
    elif corruption == "stale_source":
        metadata["source_sha256"] = "0" * 64
    elif corruption == "wrong_protocol":
        metadata["protocol"]["protocol_sha256"] = "0" * 64
    elif corruption == "wrong_seed":
        metadata["seed"] = 99
    else:
        metadata["runtime"]["train_device"] = "wrong-device"
    (run / "metadata.json").write_text(json.dumps(metadata))
    report = case.confirm()
    assert not report["checks_passed"]
    assert check in _checks(report)


@pytest.mark.parametrize(
    "corruption,check",
    [
        ("missing_training_metric", "required_metric_schedule"),
        ("prey_nan", "all_groups_finite"),
        ("duplicate", "unique_metric_records"),
        ("missing_evaluation", "required_metric_schedule"),
        ("missing_episode", "evaluation_episode_coverage"),
        ("online_reward_mismatch", "online_domain_reward_accounting"),
        ("online_terminated", "online_domain_episode_completion"),
    ],
)
def test_metric_coverage_finiteness_and_domain_accounting_fail_loudly(
    confirmation_case,
    corruption,
    check,
):
    case = confirmation_case
    run_id = next(iter(case.actions))
    rows = case.rows[run_id]
    if corruption == "missing_training_metric":
        rows[:] = [
            row
            for row in rows
            if not (
                row["phase"] == "training"
                and row["group"] == "adversary"
                and row["metric"] == "ESS"
            )
        ]
    elif corruption == "prey_nan":
        next(row for row in rows if row["group"] == "agent")["value"] = float("nan")
    elif corruption == "duplicate":
        rows.append(dict(next(row for row in rows if row["phase"] == "evaluation")))
    elif corruption == "missing_evaluation":
        rows[:] = [
            row for row in rows if not (row["phase"] == "evaluation" and row["frames"] == 12000)
        ]
    elif corruption == "missing_episode":
        rows[:] = [
            row
            for row in rows
            if not (
                row["phase"] == "evaluation"
                and row["metric"] == "return_episode"
                and row["sample"] == "127"
                and row["frames"] == 12000
            )
        ]
    elif corruption == "online_reward_mismatch":
        next(
            row
            for row in rows
            if row["phase"] == "evaluation_domain_episode" and row["metric"] == "return"
        )["value"] = 1000.0
    else:
        next(
            row
            for row in rows
            if row["phase"] == "evaluation_domain_episode" and row["metric"] == "terminated"
        )["value"] = 1.0
    _write_rows(case.suite / run_id / "metrics.csv", rows)
    report = case.confirm()
    assert not report["checks_passed"]
    assert check in _checks(report)


@pytest.mark.parametrize(
    "corruption,check",
    [
        ("missing_mode", "action_audit_coverage"),
        ("short_episode", "action_audit_coverage"),
        ("different_seed_bank", "action_audit_coverage"),
        ("source", "audit_provenance"),
        ("checkpoint", "audit_provenance"),
        ("runtime_override", "audit_runtime"),
        ("nonfinite_quantile", "action_audit_finite"),
        ("nonpositive_scale", "action_audit_positive_scale"),
    ],
)
def test_action_audits_require_full_matched_coverage_and_provenance(
    confirmation_case, corruption, check
):
    case = confirmation_case
    audit = next(iter(case.actions.values()))
    if corruption == "missing_mode":
        for name in list(audit):
            if name.startswith("random_"):
                del audit[name]
    elif corruption == "short_episode":
        audit["random_episode_transitions"][0] = 99
    elif corruption == "different_seed_bank":
        audit["random_episode_ids"][0] = "seed:0/episode:0"
    elif corruption == "source":
        audit["analysis_provenance"]["source_sha256"] = "0" * 64
    elif corruption == "checkpoint":
        audit["analysis_provenance"]["checkpoint_sha256"] = "0" * 64
    elif corruption == "runtime_override":
        audit["analysis_provenance"]["allow_runtime_mismatch"] = True
    elif corruption == "nonpositive_scale":
        audit["random_scale_quantiles"]["p05"] = 0.0
    else:
        audit["random_scale_quantiles"]["p99"] = float("nan")
    report = case.confirm()
    assert not report["checks_passed"]
    assert check in _checks(report)


@pytest.mark.parametrize(
    "corruption,check",
    [
        ("reward", "domain_reward_accounting"),
        ("coverage", "domain_audit_coverage"),
        ("aggregate", "domain_aggregate_accounting"),
        ("terminated", "domain_episode_completion"),
    ],
)
def test_domain_audits_require_consistent_episodes_and_reward(confirmation_case, corruption, check):
    case = confirmation_case
    mode = next(iter(case.domains.values()))["modes"]["RANDOM"]
    if corruption == "reward":
        mode["episodes"][0]["metrics"]["reward_accounting_max_abs_error"] = 0.1
    elif corruption == "coverage":
        mode["episodes"].pop()
    elif corruption == "aggregate":
        mode["means"]["return"] += 1
    else:
        mode["episodes"][0]["metrics"]["terminated"] = 1
    report = case.confirm()
    assert not report["checks_passed"]
    assert check in _checks(report)


def test_missing_suite_produces_explicit_nonapproval_report(tmp_path, config_root):
    report = confirm_pcp_suite(
        tmp_path / "absent",
        protocol_path=config_root / "protocols" / "pcp_corrected_v1.yaml",
        repo_root=config_root.parent,
        action_audits={},
        domain_audits={},
    )
    assert not report["checks_passed"]
    assert "required_suite" in _checks(report)
    assert report["approval_decision"] == "not_issued"


@pytest.mark.parametrize("field,value", [("frames", 120000), ("seed", 9), ("group_policies", {})])
def test_matching_audit_digest_cannot_hide_wrong_final_checkpoint(confirmation_case, field, value):
    case = confirmation_case
    run_id = next(iter(case.actions))
    checkpoint = case.suite / run_id / "checkpoints" / "policy_state.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload[field] = value
    torch.save(payload, checkpoint)
    # Both audit documents honestly identify these bytes: it is the checkpoint
    # content that is wrong, not a stale checksum.
    for audit in (case.actions[run_id], case.domains[run_id]):
        audit["analysis_provenance"]["checkpoint_sha256"] = file_sha256(checkpoint)
    report = case.confirm()
    assert not report["checks_passed"]
    expected = (
        "final_checkpoint_groups" if field == "group_policies" else "final_checkpoint_identity"
    )
    assert expected in _checks(report)


def test_matching_audits_cannot_hide_old_time_limit_semantics(confirmation_case):
    case = confirmation_case
    run_id = next(iter(case.actions))
    path = case.suite / run_id / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["task_runtime_contract"]["time_limit_protocol"] = "terminated_on_timeout"
    path.write_text(json.dumps(metadata))
    for audit in (case.actions[run_id], case.domains[run_id]):
        audit["analysis_provenance"]["task_runtime_contract"] = metadata["task_runtime_contract"]
    report = case.confirm()
    assert not report["checks_passed"]
    assert "saved_task_contract" in _checks(report)


def test_confirmation_cli_writes_failed_review_report_for_missing_inputs(tmp_path, config_root):
    spec = importlib.util.spec_from_file_location(
        "confirmation_cli", config_root.parent / "scripts" / "confirm_pcp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "confirmation_report.json"
    result = module.main(
        [
            str(tmp_path / "missing_suite"),
            "--action-audits",
            str(tmp_path / "missing_action.json"),
            "--domain-audits",
            str(tmp_path / "missing_domain.json"),
            "--out",
            str(output),
        ]
    )
    assert result == 2
    report = json.loads(output.read_text())
    assert not report["checks_passed"]
    assert report["approval_decision"] == "not_issued"
    assert report["issues"][0]["check"] == "input_artifact"
