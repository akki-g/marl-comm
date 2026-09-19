"""Artifact-level checks for the prespecified, two-seed corrected PCP confirmation.

Passing these checks means that the evidence is complete and internally
consistent enough for review. It is never an automatic protocol approval, a
learning threshold, or a claim of convergence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from commstudy.analysis.diagnostics import GATE_TRAINING_METRICS, rebuild_spec
from commstudy.experiments.config import resolved_experiment_dict, scientific_config_sha256
from commstudy.experiments.protocols import (
    content_sha256,
    load_protocol,
    protocol_spec,
    validate_built_contract,
)
from commstudy.experiments.provenance import source_fingerprint


CONFIRMATION_PROTOCOL = "pcp_confirmation_review_v1"
AUDIT_EPISODES = 32
AUDIT_BASE_SEED = 100000
EXPECTED_HORIZON = 600000
REWARD_ACCOUNTING_ATOL = 1e-5
MODES = ("DETERMINISTIC", "RANDOM")
TRAINING_HEALTH_METRICS = (
    *GATE_TRAINING_METRICS,
    "optimization_transition_count",
    "value_target_mean",
    "value_target_std",
    "value_target_scalar_count",
    "value_target_finite_fraction",
    "advantage_mean",
    "advantage_std",
    "advantage_scalar_count",
    "advantage_finite_fraction",
)
COLLECTION_HEALTH_METRICS = (
    "health_transition_count",
    "loc_scalar_count",
    "loc_finite_fraction",
    "loc_q01",
    "loc_q50",
    "loc_q99",
    "scale_scalar_count",
    "scale_finite_fraction",
    "scale_q01",
    "scale_q50",
    "scale_q99",
    "action_scalar_count",
    "action_finite_fraction",
    "action_saturated_fraction",
    "replay_log_prob_max_abs_error",
    "replay_transition_count",
)
ONLINE_DOMAIN_METRICS = (
    "episode_steps",
    "complete",
    "terminated",
    "truncated",
    "return",
    "any_contact",
    "first_contact_observed",
    "first_contact_step_or_horizon",
    "contact_duration_steps",
    "contact_pair_steps",
    "simultaneous_contact_steps",
    "reward_accounting_max_abs_error",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_digest(value, issues, label):
    if not isinstance(value, Mapping):
        return None
    try:
        return content_sha256(value)
    except (TypeError, ValueError) as error:
        _issue(issues, "audit_encoding", f"{label}: {error}")
        return None


def expected_evaluation_frames(protocol: Mapping[str, Any]) -> list[int]:
    """BenchMARL evaluates after the first update and then at its frame interval."""
    config = protocol["base_spec"]["experiment"]
    batch = config["on_policy_collected_frames_per_batch"]
    horizon = config["max_n_frames"]
    interval = config["evaluation_interval"]
    return [
        frame
        for frame in range(batch, horizon + 1, batch)
        if frame == batch or frame % interval == 0
    ]


def _issue(issues, check: str, detail: str) -> None:
    issues.append({"check": check, "detail": detail})


def _expect(issues, condition: bool, check: str, detail: str) -> bool:
    if not condition:
        _issue(issues, check, detail)
    return bool(condition)


def _close(left, right, *, atol=1e-5) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _read_json(path: Path, issues) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise TypeError("expected a JSON object")
        return value
    except (OSError, ValueError, TypeError) as error:
        _issue(issues, "required_artifact", f"{path.name}: {error}")
        return None


def _read_metrics(path: Path, issues):
    """Keep unique metric observations; identical duplicates are errors too."""
    observations = {}
    counts = defaultdict(int)
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"frames", "iteration", "phase", "group", "metric", "value", "sample"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"missing columns {sorted(required - set(reader.fieldnames or []))}"
                )
            for index, row in enumerate(reader, 2):
                try:
                    frame, iteration, value = (
                        int(row["frames"]),
                        int(row["iteration"]),
                        float(row["value"]),
                    )
                except (TypeError, ValueError) as error:
                    _issue(issues, "metric_encoding", f"CSV line {index}: {error}")
                    continue
                group = row["group"] or ""
                counts[group] += 1
                if not math.isfinite(value):
                    _issue(
                        issues,
                        "all_groups_finite",
                        f"CSV line {index}: {group}/{row['phase']}/{row['metric']} is non-finite.",
                    )
                key = (row["phase"], group, row["metric"], frame, row["sample"] or "")
                if key in observations:
                    prior = observations[key][0]
                    kind = "conflicting" if prior != value else "duplicate"
                    _issue(issues, "unique_metric_records", f"{kind} metric observation {key}.")
                    continue
                observations[key] = (value, iteration)
    except (OSError, ValueError, csv.Error) as error:
        _issue(issues, "required_artifact", f"metrics.csv: {error}")
    return observations, dict(counts)


def _series(observations, phase, group, metric):
    return {
        frame: value
        for (p, g, m, frame, sample), (value, _) in observations.items()
        if (p, g, m, sample) == (phase, group, metric, "")
    }


def _require_series(observations, issues, phase, group, metric, expected_frames):
    series = _series(observations, phase, group, metric)
    missing = sorted(set(expected_frames) - set(series))
    extra = sorted(set(series) - set(expected_frames))
    _expect(
        issues,
        not missing and not extra,
        "required_metric_schedule",
        f"{group}/{phase}/{metric}: missing frames {missing}; extra frames {extra}.",
    )
    return series


def _descriptive_returns(curve: Mapping[int, float]) -> dict[str, Any]:
    points = sorted(curve.items())
    if not points or not all(math.isfinite(value) for _, value in points):
        return {}
    values = [value for _, value in points]
    final_count = math.ceil(len(points) * 0.1)
    late_count = math.ceil(len(points) * 0.2)
    final = sum(values[-final_count:]) / final_count
    early = sum(values[:final_count]) / final_count
    area = sum(
        (x1 - x0) * (y0 + y1) / 2
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=True)
    )
    span = points[-1][0] - points[0][0]
    return {
        "evaluation_frames": [frame for frame, _ in points],
        "evaluation_returns": values,
        "first_logged_evaluation_frames": points[0][0],
        "first_logged_evaluation_return": values[0],
        "initial_reference": "first_logged_evaluation_after_first_optimizer_update",
        "final_window_points": final_count,
        "final_window_frames": [frame for frame, _ in points[-final_count:]],
        "final_window_mean_return": final,
        "early_window_frames": [frame for frame, _ in points[:final_count]],
        "early_window_mean": early,
        "late_window_mean": final,
        "early_late_delta": final - early,
        "final_evaluation_return": values[-1],
        "late_twenty_percent_points": late_count,
        "late_twenty_percent_mean_return": sum(values[-late_count:]) / late_count,
        "final_minus_first_logged_evaluation": final - values[0],
        "auc": area,
        "auc_frame_interval": [points[0][0], points[-1][0]],
        "normalized_auc": area / span if span else None,
    }


def _validate_online_domain(observations, issues, frames, eval_frames, episodes, steps, batch):
    for metric in (
        "transition_count",
        "reward_accounting_max_abs_error",
        "contact_pairs_mean",
        "predator_reward_mean",
    ):
        _require_series(observations, issues, "collection_domain", "adversary", metric, frames)
    for frame in frames:

        def scalar(name, frame=frame):
            return observations.get(
                ("collection_domain", "adversary", name, frame, ""), (None, None)
            )[0]

        error = scalar("reward_accounting_max_abs_error")
        _expect(
            issues,
            scalar("transition_count") == batch
            and isinstance(error, (int, float))
            and 0 <= error <= REWARD_ACCOUNTING_ATOL,
            "collection_domain_accounting",
            f"Frame {frame}: missing collection coverage or incorrect contact reward.",
        )
        pairs, reward = scalar("contact_pairs_mean"), scalar("predator_reward_mean")
        _expect(
            issues,
            isinstance(pairs, (int, float)) and _close(reward, 10 * pairs),
            "collection_domain_accounting",
            f"Frame {frame}: collection reward/contact mismatch.",
        )
    by_frame = defaultdict(lambda: defaultdict(dict))
    for (phase, group, metric, frame, sample), (value, _) in observations.items():
        if phase == "evaluation_domain_episode" and group == "adversary":
            by_frame[frame][sample][metric] = value
    curves = defaultdict(dict)
    for frame in eval_frames:
        rows = by_frame[frame]
        _expect(
            issues,
            set(rows) == {str(index) for index in range(episodes)},
            "online_domain_episode_coverage",
            f"Frame {frame}: missing/extra domain episodes.",
        )
        complete = []
        for sample, metrics in rows.items():
            if not _expect(
                issues,
                set(ONLINE_DOMAIN_METRICS).issubset(metrics),
                "online_domain_required_metrics",
                f"Frame {frame}/episode {sample}: incomplete domain metrics.",
            ):
                continue
            complete.append(metrics)
            _expect(
                issues,
                metrics["episode_steps"] == steps
                and metrics["complete"] == 1
                and metrics["terminated"] == 0
                and metrics["truncated"] == 1,
                "online_domain_episode_completion",
                f"Frame {frame}/episode {sample}: "
                "incomplete or incorrectly terminated time-limit episode.",
            )
            value = observations.get(
                ("evaluation", "", "return_episode", frame, sample), (None, None)
            )[0]
            _expect(
                issues,
                _close(metrics["return"], value)
                and _close(
                    metrics["return"],
                    10 * metrics["contact_pair_steps"],
                    atol=steps * REWARD_ACCOUNTING_ATOL,
                )
                and 0 <= metrics["reward_accounting_max_abs_error"] <= REWARD_ACCOUNTING_ATOL,
                "online_domain_reward_accounting",
                f"Frame {frame}/episode {sample}: "
                "normal episode return, domain return, and contact exposure disagree.",
            )
            duration = metrics["contact_duration_steps"]
            _expect(
                issues,
                0 <= metrics["simultaneous_contact_steps"] <= duration <= steps
                and metrics["contact_pair_steps"] >= duration
                and metrics["any_contact"] == int(duration > 0)
                and metrics["first_contact_observed"] == int(duration > 0),
                "online_domain_contact_accounting",
                f"Frame {frame}/episode {sample}: contact counts or observed flags disagree.",
            )
            first = metrics["first_contact_step_or_horizon"]
            _expect(
                issues,
                1 <= first <= steps and (duration > 0 or first == steps),
                "online_domain_first_contact",
                f"Frame {frame}/episode {sample}: invalid censored first-contact time.",
            )
        if len(complete) != episodes:
            continue
        for metric in ONLINE_DOMAIN_METRICS:
            expected = (
                max(row[metric] for row in complete)
                if metric == "reward_accounting_max_abs_error"
                else sum(row[metric] for row in complete) / episodes
            )
            actual = observations.get(
                ("evaluation_domain", "adversary", metric, frame, ""), (None, None)
            )[0]
            _expect(
                issues,
                _close(actual, expected),
                "online_domain_aggregate_accounting",
                f"Frame {frame}/{metric}: aggregate does not match the episode records.",
            )
            curves[metric][frame] = expected
        for metric, expected in {
            "episode_count": episodes,
            "complete_episodes": episodes,
            "transition_count": episodes * steps,
        }.items():
            actual = observations.get(
                ("evaluation_domain", "adversary", metric, frame, ""), (None, None)
            )[0]
            _expect(
                issues,
                actual == expected,
                "online_domain_aggregate_coverage",
                f"Frame {frame}/{metric}: wrong denominator.",
            )
    result = {}
    count = math.ceil(len(eval_frames) * 0.1)
    for metric in (
        "any_contact",
        "contact_duration_steps",
        "contact_pair_steps",
        "first_contact_step_or_horizon",
    ):
        curve = curves[metric]
        if set(curve) == set(eval_frames) and all(math.isfinite(value) for value in curve.values()):
            first, late = curve[eval_frames[0]], sum(curve[f] for f in eval_frames[-count:]) / count
            early = sum(curve[f] for f in eval_frames[:count]) / count
            result[metric] = {
                "first_logged_evaluation": first,
                "final_window_mean": late,
                "final_minus_first_logged_evaluation": late - first,
                "early_window_mean": early,
                "late_window_mean": late,
                "early_late_delta": late - early,
                "window_points": count,
            }
    return result


def _validate_provenance(provenance, metadata, run_dir, expected_hash, source_hash, issues, label):
    if not _expect(
        issues,
        isinstance(provenance, Mapping),
        "audit_provenance",
        f"{label}: missing analysis provenance.",
    ):
        return
    expected = {
        "scientific_config_sha256": expected_hash,
        "source_sha256": source_hash,
        "resolved_config_sha256": file_sha256(run_dir / "resolved_config.yaml"),
        "checkpoint_sha256": file_sha256(run_dir / "checkpoints" / "policy_state.pt"),
        "task_runtime_contract": metadata.get("task_runtime_contract"),
        "versions": metadata.get("versions"),
    }
    for key, value in expected.items():
        _expect(
            issues,
            provenance.get(key) == value,
            "audit_provenance",
            f"{label}: {key} does not match the saved/current run artifacts.",
        )
    _expect(
        issues,
        provenance.get("allow_runtime_mismatch") is False
        and provenance.get("compatibility_mismatches") == {},
        "audit_runtime",
        f"{label}: cross-runtime or incompatible re-evaluation cannot confirm this protocol.",
    )
    source_run = provenance.get("source_run")
    _expect(
        issues,
        isinstance(source_run, str) and Path(source_run).resolve() == run_dir.resolve(),
        "audit_provenance",
        f"{label}: source_run does not name this run.",
    )
    overrides = provenance.get("analysis_overrides") or {}
    for key in ("train_device", "sampling_device", "buffer_device"):
        if key in overrides:
            _expect(
                issues,
                overrides[key] == metadata.get("runtime", {}).get(key),
                "audit_runtime",
                f"{label}: {key} changed during the audit.",
            )


def _validate_action_audit(
    audit, metadata, run_dir, expected_hash, source_hash, steps, n_predators, issues
):
    if not _expect(
        issues,
        isinstance(audit, Mapping),
        "required_action_audit",
        "Missing frozen-policy action audit.",
    ):
        return
    _expect(
        issues,
        audit.get("run_id") == run_dir.name,
        "action_audit_identity",
        "Action audit names another run.",
    )
    _expect(
        issues,
        audit.get("measured_group") == "adversary" and audit.get("policy_audited") is True,
        "action_audit_group",
        "Action audit must measure the frozen predator actor.",
    )
    _validate_provenance(
        audit.get("analysis_provenance"),
        metadata,
        run_dir,
        expected_hash,
        source_hash,
        issues,
        "action",
    )
    ids = [f"seed:{AUDIT_BASE_SEED}/episode:{index}" for index in range(AUDIT_EPISODES)]
    for mode in MODES:
        prefix = mode.lower()
        for field, expected in {
            "episodes": AUDIT_EPISODES,
            "steps": steps,
            "complete_episodes": AUDIT_EPISODES,
            "group": "adversary",
            "episode_ids": ids,
            "episode_transitions": [steps] * AUDIT_EPISODES,
            "saturation_denominator": "finite_action_scalars",
            "saturation_threshold": 0.999,
            "finite_action_fraction": 1.0,
            "finite_location_fraction": 1.0,
            "finite_scale_fraction": 1.0,
            "action_scalars": AUDIT_EPISODES * steps * n_predators * 2,
            "finite_action_scalars": AUDIT_EPISODES * steps * n_predators * 2,
        }.items():
            actual = audit.get(f"{prefix}_{field}")
            if isinstance(actual, tuple):
                actual = list(actual)
            _expect(
                issues,
                actual == expected,
                "action_audit_coverage",
                f"{mode}/{field}: expected {expected!r}, got {actual!r}.",
            )
        for field in ("action_scalars", "finite_action_scalars"):
            _expect(
                issues,
                type(audit.get(f"{prefix}_{field}")) is int and audit[f"{prefix}_{field}"] > 0,
                "action_audit_coverage",
                f"{mode}/{field} must be a positive scalar denominator.",
            )
        _expect(
            issues,
            audit.get(f"{prefix}_finite_action_scalars") == audit.get(f"{prefix}_action_scalars"),
            "action_audit_finite",
            f"{mode}: not all action scalars were finite.",
        )
        for field in (
            "mean_abs_action",
            "max_abs_action",
            "saturated_action_fraction",
            "mean_abs_location",
            "max_abs_location",
            "mean_scale",
            "max_scale",
        ):
            value = audit.get(f"{prefix}_{field}")
            _expect(
                issues,
                isinstance(value, (int, float)) and math.isfinite(value),
                "action_audit_finite",
                f"{mode}/{field} is missing or non-finite.",
            )
        for field in ("action_abs_quantiles", "location_abs_quantiles", "scale_quantiles"):
            quantiles = audit.get(f"{prefix}_{field}")
            _expect(
                issues,
                isinstance(quantiles, Mapping)
                and set(quantiles) == {"p05", "p50", "p95", "p99"}
                and all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in quantiles.values()
                ),
                "action_audit_finite",
                f"{mode}/{field}: missing or non-finite quantiles.",
            )
        scale_quantiles = audit.get(f"{prefix}_scale_quantiles")
        scale_values = [audit.get(f"{prefix}_mean_scale"), audit.get(f"{prefix}_max_scale")]
        if isinstance(scale_quantiles, Mapping):
            scale_values.extend(scale_quantiles.values())
        _expect(
            issues,
            all(
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in scale_values
            ),
            "action_audit_positive_scale",
            f"{mode}: distribution scale summaries must be strictly positive.",
        )


def _validate_domain_audit(
    audit, metadata, run_dir, expected_hash, source_hash, steps, n_predators, n_prey, issues
):
    # The domain collector provides the detailed per-episode schema. This
    # adapter validates its coverage and reward accounting independently.
    if not _expect(
        issues,
        isinstance(audit, Mapping),
        "required_domain_audit",
        "Missing frozen-policy PCP domain audit.",
    ):
        return
    from commstudy.analysis.pcp_domain import summarize_domain_episodes
    from commstudy.tasks.vmas.domain_metrics import PCP_DOMAIN_PROTOCOL

    _expect(
        issues,
        type(audit.get("schema_version")) is int
        and audit["schema_version"] == 1
        and audit.get("domain_protocol") == PCP_DOMAIN_PROTOCOL
        and audit.get("group") == "adversary",
        "domain_audit_schema",
        "Wrong PCP domain protocol/schema or measured group.",
    )
    _validate_provenance(
        audit.get("analysis_provenance"),
        metadata,
        run_dir,
        expected_hash,
        source_hash,
        issues,
        "domain",
    )
    modes = audit.get("modes", {})
    if not _expect(
        issues,
        isinstance(modes, Mapping) and set(modes) == set(MODES),
        "domain_audit_coverage",
        "Domain audit must contain exactly both exploration modes.",
    ):
        return
    required = {
        "episode_steps",
        "complete",
        "terminated",
        "truncated",
        "return",
        "any_contact",
        "first_contact_observed",
        "first_contact_step_or_horizon",
        "contact_duration_steps",
        "contact_pair_steps",
        "simultaneous_contact_steps",
        "any_simultaneous_contact",
        "contacting_predators_mean",
        "contacting_predators_max",
        "nearest_predator_prey_distance_mean",
        "nearest_predator_prey_distance_min",
        "mean_predator_nearest_prey_distance",
        "predator_boundary_fraction",
        "prey_boundary_fraction",
        "predator_obstacle_collision_pair_steps",
        "predator_teammate_collision_pair_steps",
        "sender_visible_receiver_blind_pairs_mean",
        "sender_visible_receiver_blind_prey_triples_mean",
        "reward_accounting_max_abs_error",
        *{f"prey_visible_to_{k}_predators_mean" for k in range(n_predators + 1)},
    }
    for mode in MODES:
        summary = modes[mode]
        rows = summary.get("episodes", [])
        if not _expect(
            issues,
            isinstance(rows, list) and len(rows) == AUDIT_EPISODES,
            "domain_audit_coverage",
            f"{mode}: require exactly {AUDIT_EPISODES} episodes.",
        ):
            continue
        valid = []
        for index, row in enumerate(rows):
            _expect(
                issues,
                row.get("episode_id") == f"seed:{AUDIT_BASE_SEED}/episode:{index}"
                and row.get("seed") == AUDIT_BASE_SEED + index,
                "domain_episode_identity",
                f"{mode}/episode {index}: audit seed or ID differs from the declared bank.",
            )
            metrics = row.get("metrics")
            if not _expect(
                issues,
                isinstance(metrics, Mapping)
                and required.issubset(metrics)
                and all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in metrics.values()
                ),
                "domain_required_metrics",
                f"{mode}/episode {index}: missing/non-finite PCP domain metrics.",
            ):
                continue
            valid.append(row)
            _expect(
                issues,
                metrics["episode_steps"] == steps
                and metrics["complete"] == 1
                and metrics["terminated"] == 0
                and metrics["truncated"] == 1,
                "domain_episode_completion",
                f"{mode}/episode {index}: incomplete/wrong time limit.",
            )
            _expect(
                issues,
                0 <= metrics["reward_accounting_max_abs_error"] <= REWARD_ACCOUNTING_ATOL
                and _close(
                    metrics["return"],
                    10 * metrics["contact_pair_steps"],
                    atol=steps * REWARD_ACCOUNTING_ATOL,
                ),
                "domain_reward_accounting",
                f"{mode}/episode {index}: shared unshaped reward does not equal contact exposure.",
            )
            duration = metrics["contact_duration_steps"]
            simultaneous = metrics["simultaneous_contact_steps"]
            _expect(
                issues,
                0 <= simultaneous <= duration <= steps
                and duration <= metrics["contact_pair_steps"] <= n_predators * n_prey * steps,
                "domain_count_bounds",
                f"{mode}/episode {index}: impossible contact counts.",
            )
            observed = int(duration > 0)
            _expect(
                issues,
                metrics["any_contact"] == observed
                and metrics["first_contact_observed"] == observed
                and metrics["any_simultaneous_contact"] == int(simultaneous > 0),
                "domain_contact_accounting",
                f"{mode}/episode {index}: contact flags disagree.",
            )
            first = metrics["first_contact_step_or_horizon"]
            _expect(
                issues,
                1 <= first <= steps and (observed or first == steps),
                "domain_first_contact",
                f"{mode}/episode {index}: invalid censored contact time.",
            )
            visibility = [
                metrics[f"prey_visible_to_{k}_predators_mean"] for k in range(n_predators + 1)
            ]
            _expect(
                issues,
                all(0 <= value <= n_prey for value in visibility)
                and _close(sum(visibility), n_prey),
                "domain_visibility_accounting",
                f"{mode}/episode {index}: visibility bins do not account for every prey.",
            )
        if len(valid) != AUDIT_EPISODES:
            continue
        recomputed = summarize_domain_episodes(valid)
        for key, expected in recomputed.items():
            actual = summary.get(key)
            if isinstance(expected, Mapping):
                equal = (
                    isinstance(actual, Mapping)
                    and set(actual) == set(expected)
                    and all(_close(actual[name], value) for name, value in expected.items())
                )
            else:
                equal = actual is None if expected is None else _close(actual, expected)
            _expect(
                issues,
                equal,
                "domain_aggregate_accounting",
                f"{mode}/{key}: saved aggregate differs from its per-episode records.",
            )


def _validate_run(
    run_dir, metadata, protocol, protocol_hash, source_hash, action, domain, *, condition=None,
):
    issues = []
    condition = condition or {
        "model": "pcp_comm_identity", "ablation": "main", "ablation_value": None,
        "stage": "confirmation",
    }
    result = {
        "run_id": run_dir.name, "seed": metadata.get("seed"), "issues": issues, **condition,
    }
    from commstudy.experiments.protocols import validate_protocol_condition
    try:
        validate_protocol_condition(protocol, seed=metadata.get("seed"), **condition)
    except (ValueError, KeyError, TypeError) as error:
        _issue(issues, "declared_condition", str(error))
        result["validation_passed"] = False
        return result
    status = _read_json(run_dir / "status.json", issues)
    _expect(
        issues,
        metadata.get("status") == "completed"
        and status is not None
        and status.get("status") == "completed",
        "completed_run",
        "Both managed metadata and status must be completed.",
    )
    for key, expected in {
        "run_id": run_dir.name,
        "suite_id": run_dir.parent.name,
        "task": "vmas_predator_capture_prey",
        "algorithm": "mappo",
        "model": condition["model"],
        "execution_purpose": "scientific",
        "ablation": condition["ablation"],
        "ablation_value": condition["ablation_value"],
        "frames": EXPECTED_HORIZON,
    }.items():
        _expect(
            issues,
            metadata.get(key) == expected,
            "run_identity",
            f"{key}: expected {expected!r}, got {metadata.get(key)!r}.",
        )
    _expect(
        issues,
        metadata.get("source_sha256") == source_hash,
        "current_source",
        "Saved research source differs from the current confirmation source.",
    )
    try:
        validate_built_contract(
            protocol,
            {
                "task_contract": metadata.get("task_runtime_contract") or {},
                "training_groups": sorted(protocol["contracts"]["training_groups"]),
            },
        )
    except (ValueError, KeyError, TypeError) as error:
        _issue(issues, "saved_task_contract", str(error))
    saved_protocol = metadata.get("protocol") or {}
    _expect(
        issues,
        saved_protocol.get("factors") == {
            "model": condition["model"], condition["ablation"]: condition["ablation_value"],
        },
        "protocol_factors",
        "Saved protocol factor labels differ from the declared condition.",
    )
    for key, expected in {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "source_sha256": source_hash,
        "stage": condition["stage"],
    }.items():
        _expect(
            issues,
            saved_protocol.get(key) == expected,
            "protocol_binding",
            f"Saved protocol binding {key} differs from the candidate.",
        )
    _expect(
        issues,
        isinstance(saved_protocol.get("approval_sha256"), str)
        and len(saved_protocol["approval_sha256"]) == 64,
        "protocol_binding",
        "Saved confirmation launch approval digest is missing.",
    )
    runtime = protocol["runtime"]
    for key in ("train_device", "sampling_device", "buffer_device"):
        _expect(
            issues,
            str(metadata.get("runtime", {}).get(key)).split(":")[0] == runtime["device"],
            "saved_runtime",
            f"Saved {key} differs from the protocol.",
        )
    _expect(
        issues,
        metadata.get("runtime", {}).get("torch_num_threads") == runtime["threads"],
        "saved_runtime",
        "Saved thread count differs from the protocol.",
    )
    versions = {"python": runtime["python"], **runtime["packages"]}
    for name, version in versions.items():
        _expect(
            issues,
            metadata.get("versions", {}).get(name) == version,
            "saved_runtime",
            f"Saved {name} version differs from the protocol.",
        )
    try:
        spec = rebuild_spec(run_dir)
        expected_spec = protocol_spec(
            protocol, model=condition["model"], seed=metadata["seed"],
            ablation=condition["ablation"], ablation_value=condition["ablation_value"],
        )
        actual, expected = resolved_experiment_dict(spec), resolved_experiment_dict(expected_spec)
        actual["experiment"].pop("save_folder", None)
        expected["experiment"].pop("save_folder", None)
        _expect(
            issues,
            actual == expected,
            "resolved_protocol",
            "Full resolved configuration differs from the declared protocol/seed.",
        )
        expected_hash = scientific_config_sha256(expected_spec)
        _expect(
            issues,
            metadata.get("scientific_config_sha256") == expected_hash,
            "scientific_configuration_hash",
            "Saved scientific hash differs from the protocol.",
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        _issue(issues, "resolved_protocol", str(error))
        result["validation_passed"] = False
        return result
    configuration = expected_spec.experiment
    batch = configuration["on_policy_collected_frames_per_batch"]
    steps = expected_spec.task_config["params"]["max_steps"]
    frames = list(range(batch, EXPECTED_HORIZON + 1, batch))
    eval_frames = expected_evaluation_frames(protocol)
    _expect(
        issues,
        metadata.get("iterations") == len(frames),
        "completed_horizon",
        "Saved optimizer iteration count does not cover the complete horizon.",
    )
    observations, group_counts = _read_metrics(run_dir / "metrics.csv", issues)
    result["logged_scalars_by_group"] = group_counts
    groups = protocol["contracts"]["training_groups"]
    for group in groups:
        for name in TRAINING_HEALTH_METRICS:
            series = _require_series(observations, issues, "training", group, name, frames)
            if name.endswith("_finite_fraction"):
                _expect(
                    issues,
                    all(value == 1 for value in series.values()),
                    "training_finite",
                    f"{group}/{name} includes non-finite samples.",
                )
        for name in COLLECTION_HEALTH_METRICS:
            series = _require_series(observations, issues, "collection", group, name, frames)
            if name.endswith("_finite_fraction"):
                _expect(
                    issues,
                    all(value == 1 for value in series.values()),
                    "collection_finite",
                    f"{group}/{name} includes non-finite samples.",
                )
            if name in {"health_transition_count", "replay_transition_count"}:
                _expect(
                    issues,
                    all(value == batch for value in series.values()),
                    "collection_transition_coverage",
                    f"{group}/{name}: wrong denominator.",
                )
    for (phase, group, metric, frame, _), (_, iteration) in observations.items():
        if phase in {
            "collection",
            "training",
            "evaluation",
            "collection_domain",
            "evaluation_domain",
            "evaluation_domain_episode",
        }:
            _expect(
                issues,
                frame in (eval_frames if phase.startswith("evaluation") else frames)
                and iteration == frame // batch - 1,
                "metric_frame_iteration",
                f"{phase}/{group}/{metric}: unexpected frame/iteration {frame}/{iteration}.",
            )
    curve = _require_series(
        observations, issues, "evaluation", "adversary", "return_mean", eval_frames
    )
    study = _require_series(observations, issues, "evaluation", "", "return_mean", eval_frames)
    lengths = _require_series(
        observations, issues, "evaluation", "", "episode_length_mean", eval_frames
    )
    transitions = _require_series(
        observations, issues, "evaluation", "adversary", "health_transition_count", eval_frames
    )
    episodes = configuration["evaluation_episodes"]
    for frame in eval_frames:
        samples = {
            sample: value
            for (phase, group, name, f, sample), (value, _) in observations.items()
            if (phase, group, name, f) == ("evaluation", "", "return_episode", frame)
        }
        _expect(
            issues,
            set(samples) == {str(index) for index in range(episodes)},
            "evaluation_episode_coverage",
            f"Frame {frame}: require exactly {episodes} unique episodes.",
        )
        if samples:
            _expect(
                issues,
                _close(sum(samples.values()) / len(samples), curve.get(frame)),
                "evaluation_return_accounting",
                f"Frame {frame}: episode/aggregate return differs.",
            )
        _expect(
            issues,
            _close(curve.get(frame), study.get(frame)),
            "measured_return_group",
            f"Frame {frame}: study/predator return differs.",
        )
        _expect(
            issues,
            lengths.get(frame) == steps and transitions.get(frame) == episodes * steps,
            "evaluation_transition_coverage",
            f"Frame {frame}: evaluation transitions are incomplete.",
        )
    result["performance"] = _descriptive_returns(curve)
    result["domain_learning_summary"] = _validate_online_domain(
        observations, issues, frames, eval_frames, episodes, steps, batch
    )
    checkpoint = run_dir / "checkpoints" / "policy_state.pt"
    if checkpoint.is_file():
        try:
            saved_policy = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for key, expected in {
                "schema_version": 1,
                "run_id": run_dir.name,
                "suite_id": run_dir.parent.name,
                "task": expected_spec.task,
                "algorithm": expected_spec.algorithm,
                "model": expected_spec.model,
                "seed": expected_spec.seed,
                "frames": EXPECTED_HORIZON,
            }.items():
                _expect(
                    issues,
                    isinstance(saved_policy, Mapping) and saved_policy.get(key) == expected,
                    "final_checkpoint_identity",
                    f"Final checkpoint {key} differs from the run.",
                )
            _expect(
                issues,
                isinstance(saved_policy, Mapping)
                and isinstance(saved_policy.get("group_policies"), Mapping)
                and set(saved_policy["group_policies"]) == set(groups),
                "final_checkpoint_groups",
                "Final checkpoint omits or adds a policy group.",
            )
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            _issue(issues, "final_checkpoint_encoding", str(error))
        try:
            task = expected_spec.task_config["params"]
            _validate_action_audit(
                action,
                metadata,
                run_dir,
                expected_hash,
                source_hash,
                steps,
                task["num_adversaries"],
                issues,
            )
            _validate_domain_audit(
                domain,
                metadata,
                run_dir,
                expected_hash,
                source_hash,
                steps,
                task["num_adversaries"],
                task["num_good_agents"],
                issues,
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            _issue(issues, "audit_schema", str(error))
    else:
        _issue(issues, "required_artifact", "Missing final frozen-policy checkpoint.")
    result["artifacts"] = {
        name: file_sha256(run_dir / name)
        for name in (
            "metadata.json",
            "status.json",
            "resolved_config.yaml",
            "metrics.csv",
            "checkpoints/policy_state.pt",
        )
        if (run_dir / name).is_file()
    }
    result["action_audit_sha256"] = _audit_digest(action, issues, "action")
    result["domain_audit_sha256"] = _audit_digest(domain, issues, "domain")
    result["validation_passed"] = not issues
    return result


def confirm_pcp_suite(
    suite_dir: Path,
    *,
    protocol_path: Path,
    repo_root: Path,
    action_audits: Mapping[str, Mapping[str, Any]],
    domain_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate evidence, summarize outcomes, and require a human research review."""
    protocol = load_protocol(protocol_path)
    protocol_hash, source_hash = content_sha256(protocol), source_fingerprint(repo_root)
    seeds = protocol["selection"]["confirmation_seeds"]
    issues = []
    _expect(
        issues,
        len(seeds) == 2,
        "confirmation_allocation",
        "This bounded confirmation requires exactly two declared fresh training seeds.",
    )
    config = protocol["base_spec"]["experiment"]
    _expect(
        issues,
        config["max_n_frames"] == EXPECTED_HORIZON and config["evaluation_episodes"] == 128,
        "confirmation_budget",
        "The bounded confirmation requires 600000 frames and 128 evaluation episodes.",
    )
    suite_dir = Path(suite_dir).resolve()
    records = []
    if suite_dir.is_dir():
        for path in sorted(suite_dir.iterdir()):
            if path.is_dir() and (path / "metadata.json").exists():
                metadata = _read_json(path / "metadata.json", issues)
                if metadata is not None:
                    records.append((path, metadata))
    else:
        _issue(issues, "required_suite", f"Suite directory is missing: {suite_dir}")
    actual_seeds = [metadata.get("seed") for _, metadata in records]
    _expect(
        issues,
        len(records) == len(seeds) and sorted(actual_seeds, key=str) == sorted(seeds, key=str),
        "exact_seed_coverage",
        f"Expected one run each for seeds {seeds}; found {actual_seeds}.",
    )
    run_ids = {path.name for path, _ in records}
    for label, audits in [("action", action_audits), ("domain", domain_audits)]:
        _expect(
            issues,
            set(audits) == run_ids,
            "audit_run_coverage",
            f"{label}: missing {sorted(run_ids - set(audits))}; "
            f"extra {sorted(set(audits) - run_ids)}.",
        )
    runs = []
    for path, metadata in records:
        try:
            runs.append(
                _validate_run(
                    path,
                    metadata,
                    protocol,
                    protocol_hash,
                    source_hash,
                    action_audits.get(path.name),
                    domain_audits.get(path.name),
                )
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            runs.append(
                {
                    "run_id": path.name,
                    "seed": metadata.get("seed"),
                    "validation_passed": False,
                    "issues": [{"check": "artifact_schema", "detail": str(error)}],
                }
            )
    passed = not issues and all(run["validation_passed"] for run in runs)
    valid_performance = [run["performance"] for run in runs if run.get("performance")]
    summary = {}
    if len(valid_performance) == len(seeds) and all(
        row.get("normalized_auc") is not None for row in valid_performance
    ):
        for metric in (
            "final_window_mean_return",
            "normalized_auc",
            "final_minus_first_logged_evaluation",
            "early_window_mean",
            "late_window_mean",
            "early_late_delta",
        ):
            summary[f"seed_mean_{metric}"] = sum(row[metric] for row in valid_performance) / len(
                seeds
            )
    return {
        "schema_version": 1,
        "measurement_protocol": CONFIRMATION_PROTOCOL,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "source_sha256": source_hash,
        "suite_dir": str(suite_dir),
        "expected_seeds": seeds,
        "expected_evaluation_frames": expected_evaluation_frames(protocol),
        "checks_passed": passed,
        "review_status": "required" if passed else "blocked_incomplete_or_invalid",
        "approval_decision": "not_issued",
        "budget_claim": protocol["selection"]["budget_claim"],
        "interpretation": (
            "Descriptive two-seed fixed-compute evidence; "
            "no convergence or automatic learning verdict."
        ),
        "issues": issues,
        "runs": runs,
        "summary": summary,
    }
