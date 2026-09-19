"""Audit and aggregate the fixed 40-policy PCP visibility/budget comparison.

This script reads completed evidence only. It never launches training, selects
checkpoints, drops an unsuccessful seed, or treats episodes as trained policies.
The deadline, target, seed allocation and bootstrap rule are fixed by the D4 plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commstudy.analysis.pcp_budget import (
    BUDGET_EVALUATION_PROTOCOL,
    INTERVENTIONS,
    paired_episode_differences,
    paired_seed_interval,
    tensor_sha256,
)
from commstudy.experiments.config import experiment_spec_from_dict, scientific_config_sha256
from commstudy.experiments.protocols import content_sha256, load_protocol, protocol_spec
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.sweeps import read_manifest


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0", "pcp_broadcast_k3")
VISIBILITIES = ("radius1", "global")
TRAINING_SEEDS = tuple(range(20, 25))
EPISODE_SEEDS = list(range(200000, 200256))
EVALUATION_FRAMES = [6000, *range(12000, 600001, 12000)]
DEADLINE, TARGET = 50, 0.80
REQUIRED_RUN_ARTIFACTS = (
    "metadata.json",
    "status.json",
    "resolved_config.yaml",
    "metrics.csv",
    "checkpoints/policy_state.pt",
)
DOMAIN_KEYS = {
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
    "contacting_predators_max",
    "contacting_predators_mean",
    "mean_predator_nearest_prey_distance",
    "nearest_predator_prey_distance_mean",
    "nearest_predator_prey_distance_min",
    "predator_boundary_fraction",
    "prey_boundary_fraction",
    "predator_obstacle_collision_pair_steps",
    "predator_teammate_collision_pair_steps",
    "sender_visible_receiver_blind_pairs_mean",
    "sender_visible_receiver_blind_prey_triples_mean",
    "reward_accounting_max_abs_error",
    "contact_by_deadline",
    *{f"prey_visible_to_{k}_predators_mean" for k in range(4)},
}
HEALTH_KEYS = {
    "action_scalars",
    "all_finite",
    "saturation_fraction",
    "max_abs_action",
    "max_abs_loc",
    "min_scale",
    "max_scale",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def read_json(path: Path, hashes: dict) -> Any:
    hashes[str(path.resolve())] = sha256(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite_numbers(values: dict, label: str) -> None:
    require(
        all(type(v) in (int, float, bool) and math.isfinite(v) for v in values.values()),
        f"{label} contains a non-finite or nonnumeric value",
    )


def validate_episode(row, *, seed, model, arm):
    require(
        row["episode_id"] == f"pcp-heldout:{seed}" and row["seed"] == seed,
        "Held-out episode identity or ordering differs from the fixed bank",
    )
    require(valid_digest(row["initial_physical_state_sha256"]), "Missing physical-state digest")
    require(
        set(row["exogenous_initial"]) == {"exogenous_episode_ids", "exogenous_steps"},
        "Missing exogenous episode/time identity",
    )
    require(row["exogenous_initial"]["exogenous_steps"] == [0], "Episode starts after time zero")
    episode_ids = row["exogenous_initial"]["exogenous_episode_ids"]
    require(
        isinstance(episode_ids, list)
        and len(episode_ids) == 1
        and type(episode_ids[0]) is int
        and episode_ids[0] >= 0,
        "Invalid exogenous episode identity",
    )
    metrics, health, costs = row["metrics"], row["action_health"], row["costs"]
    require(set(metrics) == DOMAIN_KEYS, "Unexpected or missing held-out domain metrics")
    require(set(health) == HEALTH_KEYS, "Unexpected or missing predator health metrics")
    finite_numbers(metrics, "Domain metrics")
    finite_numbers(health, "Action health")
    require(
        metrics["episode_steps"] == 100
        and metrics["complete"] == 1
        and metrics["truncated"] == 1
        and metrics["terminated"] == 0,
        "Episode is not one complete 100-step time-limit truncation",
    )
    duration, pairs, simultaneous = (
        metrics[k]
        for k in ("contact_duration_steps", "contact_pair_steps", "simultaneous_contact_steps")
    )
    for name in (
        "contact_duration_steps",
        "contact_pair_steps",
        "simultaneous_contact_steps",
        "first_contact_step_or_horizon",
        "contacting_predators_max",
        "predator_obstacle_collision_pair_steps",
        "predator_teammate_collision_pair_steps",
    ):
        require(float(metrics[name]).is_integer(), f"Fractional transition count: {name}")
    require(
        0 <= simultaneous <= duration <= 100
        and duration + simultaneous <= pairs <= duration + 2 * simultaneous,
        "Impossible contact exposure",
    )
    require(
        metrics["any_contact"] == metrics["first_contact_observed"] == int(duration > 0)
        and metrics["any_simultaneous_contact"] == int(simultaneous > 0),
        "Contact flags disagree with exposure",
    )
    first = metrics["first_contact_step_or_horizon"]
    require(1 <= first <= 100 and (duration > 0 or first == 100), "Invalid first-contact censoring")
    require(duration == 0 or duration <= 101 - first, "Contact exposure precedes first contact")
    maximum = metrics["contacting_predators_max"]
    triple_contact_steps = pairs - duration - simultaneous
    expected_maximum = (
        3 if triple_contact_steps > 0 else 2 if simultaneous > 0 else 1 if duration > 0 else 0
    )
    require(
        maximum == expected_maximum
        and abs(metrics["contacting_predators_mean"] - pairs / 100) <= 1e-6,
        "Contacting-predator count disagrees with contact exposure",
    )
    require(
        metrics["contact_by_deadline"] == int(duration > 0 and first <= DEADLINE),
        "Deadline contact outcome disagrees with first contact",
    )
    require(
        metrics["reward_accounting_max_abs_error"] == 0 and metrics["return"] == 10 * pairs,
        "Repeated-contact shared reward accounting failed",
    )
    for name in ("predator_boundary_fraction", "prey_boundary_fraction"):
        require(0 <= metrics[name] <= 1, f"Invalid {name}")
    bins = [metrics[f"prey_visible_to_{k}_predators_mean"] for k in range(4)]
    require(
        all(0 <= v <= 1 for v in bins) and abs(sum(bins) - 1) <= 1e-6,
        "Visibility bins do not account for the prey",
    )
    require(
        health["action_scalars"] == 600 and health["all_finite"] is True,
        "Predator action finiteness or scalar count failed",
    )
    require(
        0 <= health["saturation_fraction"] <= 1
        and 0 < health["min_scale"] <= health["max_scale"]
        and 0 <= health["max_abs_action"] <= 1
        and health["max_abs_loc"] >= 0,
        "Invalid action bounds, scale or saturation",
    )
    width = 32 if model in {"pcp_broadcast_k0", "pcp_broadcast_k3"} else 0
    senders = (
        (0 if arm == "severed" else (2 if arm.startswith("suppress_") else 3))
        if (model == "pcp_broadcast_k3")
        else 0
    )
    expected_costs = {
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
    }
    require(costs == expected_costs, "Observed payload accounting differs from declared budget")
    stats = row["module_stats"]
    finite_numbers(stats, "Module stats")
    for key, expected in {
        "messages_per_step": senders,
        "realized_sender_bits_per_step": senders * width * 32,
        "realized_bits_per_step": 2 * senders * width * 32,
    }.items():
        require(stats.get(key) == expected, f"Module payload count disagrees: {key}")


def aggregate_episodes(rows):
    """One trained policy is the statistical unit, irrespective of episode count."""
    summary = {
        key: float(np.mean([r["metrics"][key] for r in rows])) for key in sorted(DOMAIN_KEYS)
    }
    contacted = [r for r in rows if r["metrics"]["first_contact_observed"]]
    summary["first_contact_conditional_mean"] = (
        float(np.mean([r["metrics"]["first_contact_step_or_horizon"] for r in contacted]))
        if contacted
        else None
    )
    summary.update(
        episodes=len(rows),
        transitions=sum(r["costs"]["transitions"] for r in rows),
        action_scalars=sum(r["action_health"]["action_scalars"] for r in rows),
        action_all_finite=all(r["action_health"]["all_finite"] for r in rows),
        action_saturation_fraction=float(
            np.mean([r["action_health"]["saturation_fraction"] for r in rows])
        ),
        max_abs_action=max(r["action_health"]["max_abs_action"] for r in rows),
        max_abs_loc=max(r["action_health"]["max_abs_loc"] for r in rows),
        min_scale=min(r["action_health"]["min_scale"] for r in rows),
        max_scale=max(r["action_health"]["max_scale"] for r in rows),
        sender_payload_bits_per_step=float(
            np.mean([r["costs"]["sender_payload_bits"] / 100 for r in rows])
        ),
        edge_delivery_payload_bits_per_step=float(
            np.mean([r["costs"]["edge_delivery_payload_bits"] / 100 for r in rows])
        ),
        total_sender_payload_bits=sum(r["costs"]["sender_payload_bits"] for r in rows),
        total_edge_delivery_payload_bits=sum(
            r["costs"]["edge_delivery_payload_bits"] for r in rows
        ),
    )
    return summary


def validate_row(plan, evaluation, *, protocol, protocol_hash, plan_hash, hashes):
    run = plan.output_root / plan.suite_id / plan.run_id
    metadata = read_json(run / "metadata.json", hashes)
    status = read_json(run / "status.json", hashes)
    require(metadata["status"] == status["status"] == "completed", "Managed run is not completed")
    for key, expected in {
        "run_id": plan.run_id,
        "suite_id": plan.suite_id,
        "model": plan.model,
        "seed": plan.seed,
        "task": "vmas_predator_capture_prey",
        "algorithm": "mappo",
        "ablation": "visibility",
        "ablation_value": plan.ablation_value,
        "frames": 600000,
        "iterations": 100,
        "execution_purpose": "scientific",
    }.items():
        require(metadata.get(key) == expected, f"Managed identity mismatch: {key}")
    condition = {
        "stage": "comparison",
        "model": plan.model,
        "ablation": "visibility",
        "ablation_value": plan.ablation_value,
    }
    require(
        plan.protocol is not None
        and plan.resolved_spec is not None
        and plan.model_contract is not None,
        "Manifest lacks resolved protocol/model binding",
    )
    for key, value in {**condition, "seed": plan.seed, "sha256": protocol_hash}.items():
        require(plan.protocol.get(key) == value, f"Manifest protocol mismatch: {key}")
    source = plan.protocol["source_sha256"]
    require(
        valid_digest(source) and metadata["source_sha256"] == source, "Training source mismatch"
    )
    binding = metadata["protocol"]
    require(
        binding["protocol_sha256"] == protocol_hash
        and binding["source_sha256"] == source
        and binding["stage"] == "comparison"
        and binding["factors"] == {"model": plan.model, "visibility": plan.ablation_value},
        "Managed protocol mismatch",
    )
    require(valid_digest(binding["approval_sha256"]), "Missing launch approval binding")
    spec = experiment_spec_from_dict(plan.resolved_spec)
    expected = protocol_spec(
        protocol,
        model=plan.model,
        seed=plan.seed,
        ablation="visibility",
        ablation_value=plan.ablation_value,
    )
    spec_hash = scientific_config_sha256(spec)
    require(
        spec_hash
        == plan.scientific_hash
        == metadata["scientific_config_sha256"]
        == scientific_config_sha256(expected),
        "Scientific configuration mismatch",
    )
    require(
        metadata["task_runtime_contract"] == plan.model_contract["task_contract"],
        "Task/actor/critic contract mismatch",
    )
    require(
        metadata["parameters"] == plan.model_contract["parameter_counts"],
        "Active model capacity count mismatch",
    )
    require(
        metadata["versions"]
        == {"python": protocol["runtime"]["python"], **protocol["runtime"]["packages"]},
        "Saved package mismatch",
    )
    for device in ("sampling_device", "train_device", "buffer_device"):
        require(metadata["runtime"][device] == "cpu", "Non-CPU run in CPU comparison")
    require(metadata["runtime"]["torch_num_threads"] == 1, "Non-declared compute threads")
    evaluation_status = read_json(evaluation / "status.json", hashes)
    require(evaluation_status["status"] == "completed", "Held-out audit did not complete")
    report = read_json(evaluation / "heldout_evaluation.json", hashes)
    validation = read_json(evaluation / "training_validation.json", hashes)
    require(
        report["training_validation_sha256"] == sha256(evaluation / "training_validation.json"),
        "Full-horizon validation digest mismatch",
    )
    retained = read_json(evaluation / "retained_checkpoints.json", hashes)
    require(
        report["retained_checkpoints_sha256"] == sha256(evaluation / "retained_checkpoints.json"),
        "Retained native checkpoint inventory digest differs",
    )
    require(
        retained["schema_version"] == 1 and retained["run_id"] == plan.run_id,
        "Retained checkpoint inventory has the wrong run identity",
    )
    checkpoints = retained["checkpoints"]
    require(
        len(checkpoints) == 5
        and sorted(r["frames"] for r in checkpoints) == list(range(120000, 600001, 120000)),
        "Required five native checkpoint frames were not preserved",
    )
    for checkpoint in checkpoints:
        path = Path(checkpoint["path"]).resolve()
        require(
            path.is_relative_to(run.resolve())
            and path.name == (f"checkpoint_{checkpoint['frames']}.pt")
            and checkpoint["iterations"] == (checkpoint["frames"] // 6000),
            "Invalid retained native checkpoint identity/header",
        )
        hashes[str(path)] = sha256(path)
        require(hashes[str(path)] == checkpoint["sha256"], "Retained native checkpoint changed")
    require(
        validation.get("validation_passed") is True and validation.get("issues") == [],
        "Full-horizon validation did not pass",
    )
    for key, value in {**condition, "run_id": plan.run_id, "seed": plan.seed}.items():
        require(validation.get(key) == value, f"Full-horizon validation condition mismatch: {key}")
    require(
        set(validation["artifacts"]) == set(REQUIRED_RUN_ARTIFACTS),
        "Full-horizon validation artifact set is incomplete",
    )
    for name, digest in validation["artifacts"].items():
        actual = sha256(run / name)
        hashes[str((run / name).resolve())] = actual
        require(actual == digest, f"Run artifact changed after full-horizon validation: {name}")
    for name in ("action", "domain"):
        audit = read_json(evaluation / f"{name}_audit.json", hashes)
        require(
            content_sha256(audit) == validation[f"{name}_audit_sha256"],
            f"{name} audit differs from full-horizon validation",
        )
    for key, value in {
        "run_id": plan.run_id,
        "training_seed": plan.seed,
        "model": plan.model,
        "visibility": plan.ablation_value,
        "protocol_sha256": protocol_hash,
        "plan_sha256": plan_hash,
        "schema_version": 1,
        "evaluation_protocol": BUDGET_EVALUATION_PROTOCOL,
        "group": "adversary",
        "exploration": "DETERMINISTIC",
        "deadline": DEADLINE,
        "episode_seeds": EPISODE_SEEDS,
        "shuffle_definition": "frozen_live_packets_from_cyclic_next_episode_at_same_step",
        "shuffle_distribution_shift": (
            "donor live trajectories substituted into receiver closed-loop trajectories"
        ),
    }.items():
        require(report.get(key) == value, f"Held-out condition mismatch: {key}")
    provenance = report["analysis_provenance"]
    overrides = provenance.get("analysis_overrides", {})
    automatic = {
        "loggers": [],
        "create_json": False,
        "checkpoint_interval": 0,
        "checkpoint_at_end": False,
        "restore_file": None,
        "restore_map_location": None,
        "render": False,
    }
    require(
        set(overrides) == {*automatic, "save_folder"}
        and all(overrides[key] == value for key, value in automatic.items())
        and isinstance(overrides["save_folder"], str)
        and not Path(overrides["save_folder"]).resolve().is_relative_to(run.resolve()),
        "Held-out analysis used undeclared overrides or wrote inside its source run",
    )
    for key, value in {
        "source_run": str(run.resolve()),
        "source_sha256": source,
        "checkpoint_sha256": validation["artifacts"]["checkpoints/policy_state.pt"],
        "resolved_config_sha256": validation["artifacts"]["resolved_config.yaml"],
        "scientific_config_sha256": spec_hash,
        "versions": metadata["versions"],
        "task_runtime_contract": metadata["task_runtime_contract"],
        "allow_runtime_mismatch": False,
        "compatibility_mismatches": {},
    }.items():
        require(provenance.get(key) == value, f"Held-out source/runtime binding mismatch: {key}")
    performance = validation["performance"]
    require(
        performance["evaluation_frames"] == EVALUATION_FRAMES
        and len(performance["evaluation_returns"]) == 51,
        "Training curve horizon differs",
    )
    require(all(math.isfinite(v) for v in performance["evaluation_returns"]), "Non-finite returns")
    arms = INTERVENTIONS if plan.model == "pcp_broadcast_k3" else ("live", "severed")
    require(set(report["interventions"]) == set(arms), "Missing or extra intervention arms")
    for arm in arms:
        rows = report["interventions"][arm]
        require(len(rows) == 256, "Held-out episode count is not exactly 256")
        for seed, row in zip(EPISODE_SEEDS, rows, strict=True):
            validate_episode(row, seed=seed, model=plan.model, arm=arm)
        paired_episode_differences(report["interventions"]["live"], rows, "return")
        if plan.model != "pcp_broadcast_k3":
            require(
                all(
                    a["metrics"] == b["metrics"] and a["action_health"] == b["action_health"]
                    for a, b in zip(report["interventions"]["live"], rows, strict=True)
                ),
                "Zero-communication control failed its exact null intervention",
            )
    if plan.model == "pcp_broadcast_k3":
        bank_path = evaluation / "live_packet_bank.pt"
        hashes[str(bank_path.resolve())] = sha256(bank_path)
        require(
            report["packet_file_sha256"] == hashes[str(bank_path.resolve())],
            "Live packet file digest differs",
        )
        bank_record = torch.load(bank_path, map_location="cpu", weights_only=True)
        bank = bank_record["packets"]
        require(
            bank_record["episode_seeds"] == EPISODE_SEEDS
            and bank.shape == (256, 100, 1, 3, 32)
            and bank.dtype == torch.float32
            and bool(torch.isfinite(bank).all()),
            "Invalid frozen live packet bank",
        )
        require(
            tensor_sha256(bank) == report["live_packet_bank_sha256"], "Packet tensor hash differs"
        )
        for index, row in enumerate(report["interventions"]["shuffle"]):
            donor = (index + 1) % 256
            require(
                row["donor_episode_seed"] == EPISODE_SEEDS[donor]
                and row["donor_packet_sha256"] == tensor_sha256(bank[donor]),
                "Semantic shuffle donor identity or packet digest differs",
            )
    else:
        require(
            report.get("live_packet_bank_sha256") is None
            and report.get("packet_file_sha256") is None,
            "Zero control declares a packet bank",
        )
    return report, performance, metadata


def paired_contrast(name, terms, records, *, metric="contact_by_deadline"):
    """Use all five matched training seeds or publish an unavailable contrast."""
    values, absent = [], []
    for seed in TRAINING_SEEDS:
        selected = [records.get((model, visibility, seed)) for _, model, visibility, _ in terms]
        if any(row is None or not row["validation_passed"] for row in selected):
            absent.append(seed)
            continue
        reference = selected[0]["episode_records"][terms[0][3]]
        total = 0.0
        for (coefficient, _, _, arm), row in zip(terms, selected, strict=True):
            episodes = row["episode_records"][arm]
            paired_episode_differences(reference, episodes, metric)
            total += coefficient * row["arms"][arm][metric]
        values.append({"seed": seed, "value": total})
    result = {
        "contrast": name,
        "metric": metric,
        "per_seed": values,
        "missing_or_invalid_seeds": absent,
        "terms": [
            {"coefficient": c, "model": m, "visibility": v, "arm": a} for c, m, v, a in terms
        ],
    }
    result["status"] = "unavailable" if absent else "complete"
    result["interval"] = None if absent else paired_seed_interval([r["value"] for r in values])
    return result


def summarize(records):
    contrasts = []
    for visibility in VISIBILITIES:
        for baseline in ("pcp_local_capacity", "pcp_broadcast_k0", "pcp_comm_identity"):
            contrasts.append(
                paired_contrast(
                    f"broadcast_k3_minus_{baseline}/{visibility}",
                    [
                        (1, "pcp_broadcast_k3", visibility, "live"),
                        (-1, baseline, visibility, "live"),
                    ],
                    records,
                )
            )
        for arm in INTERVENTIONS[1:]:
            for metric in ("contact_by_deadline", "return"):
                contrasts.append(
                    paired_contrast(
                        f"broadcast_k3_live_minus_{arm}/{visibility}",
                        [
                            (1, "pcp_broadcast_k3", visibility, "live"),
                            (-1, "pcp_broadcast_k3", visibility, arm),
                        ],
                        records,
                        metric=metric,
                    )
                )
    contrasts.append(
        paired_contrast(
            "capacity_controlled_visibility_interaction_radius1_minus_global",
            [
                (1, "pcp_broadcast_k3", "radius1", "live"),
                (-1, "pcp_local_capacity", "radius1", "live"),
                (-1, "pcp_broadcast_k3", "global", "live"),
                (1, "pcp_local_capacity", "global", "live"),
            ],
            records,
        )
    )
    frontiers = []
    for visibility in VISIBILITIES:
        budgets = []
        for k in (0, 3):
            rows = [
                records.get((f"pcp_broadcast_k{k}", visibility, seed)) for seed in TRAINING_SEEDS
            ]
            complete = all(row is not None and row["validation_passed"] for row in rows)
            values = [
                {"seed": seed, "value": row["arms"]["live"]["contact_by_deadline"]}
                for seed, row in zip(TRAINING_SEEDS, rows, strict=True)
                if row is not None and row["validation_passed"]
            ]
            interval = paired_seed_interval([r["value"] for r in values]) if complete else None
            budgets.append(
                {
                    "sender_budget": k,
                    "per_seed": values,
                    "interval": interval,
                    "qualifies": interval["ci95_low"] >= TARGET if interval else None,
                }
            )
        complete = all(b["interval"] is not None for b in budgets)
        qualified = [b["sender_budget"] for b in budgets if b["qualifies"]]
        frontiers.append(
            {
                "visibility": visibility,
                "target": TARGET,
                "deadline": DEADLINE,
                "budgets": budgets,
                "smallest_tested_budget": min(qualified)
                if complete and qualified
                else ("not reached" if complete else "unavailable"),
                "rule": "95% lower percentile-bootstrap bound across five training seeds >= 0.80",
                "scope": (
                    "Tested feed-forward Broadcast budgets {0,3}, fixed 600k CPU; no untested bound"
                ),
            }
        )
    return contrasts, frontiers


def condition_summaries(records):
    summaries = []
    for model, visibility in itertools.product(MODELS, VISIBILITIES):
        rows = [records[(model, visibility, seed)] for seed in TRAINING_SEEDS]
        complete = all(row["validation_passed"] for row in rows)
        summary = {
            "model": model,
            "visibility": visibility,
            "status": "complete" if complete else "unavailable",
            "valid_seeds": [r["seed"] for r in rows if r["validation_passed"]],
            "means": {},
            "intervals": {},
            "interventions": {},
        }
        if complete:
            for arm in rows[0]["arms"]:
                summary["interventions"][arm] = {}
                for metric in rows[0]["arms"][arm]:
                    values = [r["arms"][arm][metric] for r in rows]
                    if all(type(v) in (int, float, bool) and math.isfinite(v) for v in values):
                        summary["interventions"][arm][metric] = float(np.mean(values))
            keys = rows[0]["arms"]["live"]
            for metric in keys:
                values = [r["arms"]["live"][metric] for r in rows]
                if all(type(v) in (int, float) and math.isfinite(v) for v in values):
                    summary["means"][metric] = float(np.mean(values))
            for metric in (
                "contact_by_deadline",
                "return",
                "contact_duration_steps",
                "simultaneous_contact_steps",
                "predator_boundary_fraction",
                "prey_boundary_fraction",
                "action_saturation_fraction",
            ):
                summary["intervals"][metric] = paired_seed_interval(
                    [r["arms"]["live"][metric] for r in rows]
                )
        summaries.append(summary)
    return summaries


def analyze(manifest: Path, evaluations: Path, *, protocol_path: Path, plan_path: Path) -> dict:
    hashes = {
        str(p.resolve()): sha256(p) for p in (manifest, protocol_path, plan_path, Path(__file__))
    }
    protocol, plan_hash = load_protocol(protocol_path), sha256(plan_path)
    protocol_hash = content_sha256(protocol)
    issues, records, groups = [], {}, defaultdict(list)
    plans = read_manifest(manifest)
    expected = set(itertools.product(MODELS, VISIBILITIES, TRAINING_SEEDS))
    for plan in plans:
        groups[(plan.model, plan.ablation_value, plan.seed)].append(plan)
    if (
        len(plans) != 40
        or set(groups) != expected
        or any(len(rows) != 1 for rows in groups.values())
    ):
        issues.append(
            "Manifest does not contain exactly the declared 40 unique model/visibility/seed rows"
        )
    if len({p.run_id for p in plans}) != len(plans) or len({p.suite_id for p in plans}) != 1:
        issues.append("Manifest has duplicate run IDs or inconsistent suite identity")
    sources = {(p.protocol or {}).get("source_sha256") for p in plans}
    if len(sources) != 1 or any(not valid_digest(value) for value in sources):
        issues.append("Manifest does not bind one valid common training source")
    require(
        protocol["protocol_id"] == "pcp_visibility_budget_cpu_v1"
        and protocol["selection"]["comparison_seeds"] == list(TRAINING_SEEDS),
        "Analyzer supports only the declared D4 CPU protocol and seed allocation",
    )
    found = defaultdict(list)
    for path in sorted(evaluations.rglob("heldout_evaluation.json")):
        try:
            heldout = read_json(path, hashes)
            found[heldout["run_id"]].append(path.parent)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"Invalid held-out report {path}: {error}")
    unexpected = set(found) - {p.run_id for p in plans}
    if unexpected:
        issues.append(f"Held-out evidence contains unplanned run IDs: {sorted(unexpected)}")
    for model, visibility, seed in sorted(expected):
        key = model, visibility, seed
        row = {
            "model": model,
            "visibility": visibility,
            "seed": seed,
            "run_id": None,
            "status": "missing",
            "validation_passed": False,
            "issues": [],
            "arms": {},
        }
        records[key] = row
        matches = groups.get(key, [])
        if len(matches) != 1:
            row["issues"].append("Missing or duplicated manifest condition")
            continue
        plan = matches[0]
        row["run_id"] = plan.run_id
        run = plan.output_root / plan.suite_id / plan.run_id
        row["run_directory"] = str(run.resolve())
        try:
            require(
                plan.ablation == "visibility"
                and plan.max_n_frames == 600000
                and plan.attempt == 0
                and plan.retry_of is None,
                "Manifest row is not the declared original 600k visibility attempt",
            )
            status = read_json(run / "status.json", hashes)
            row["managed_status"] = status["status"]
            require(status["status"] == "completed", f"Managed run status is {status['status']}")
            matches = found.get(plan.run_id, [])
            require(len(matches) == 1, "Missing or duplicated held-out evaluation report")
            report, performance, metadata = validate_row(
                plan,
                matches[0],
                protocol=protocol,
                protocol_hash=protocol_hash,
                plan_hash=plan_hash,
                hashes=hashes,
            )
            row.update(
                status="valid",
                validation_passed=True,
                performance=performance,
                arms={
                    arm: aggregate_episodes(episodes)
                    for arm, episodes in report["interventions"].items()
                },
                episode_records=report["interventions"],
                parameters=metadata["parameters"],
                evaluation_directory=str(matches[0].resolve()),
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            AttributeError,
            IndexError,
            EOFError,
            pickle.UnpicklingError,
        ) as error:
            row["status"] = row.get("managed_status", "invalid")
            if row["status"] == "completed":
                row["status"] = "invalid"
            row["issues"].append(f"{type(error).__name__}: {error}")
    # Validate physical episode pairing across every available condition and
    # training seed; identical held-out seed IDs alone are insufficient.
    reference = next((r for r in records.values() if r["validation_passed"]), None)
    if reference:
        for row in records.values():
            if row["validation_passed"]:
                try:
                    paired_episode_differences(
                        reference["episode_records"]["live"],
                        row["episode_records"]["live"],
                        "return",
                    )
                except (ValueError, KeyError) as error:
                    row.update(status="invalid", validation_passed=False)
                    row["issues"].append(str(error))
    contrasts, frontiers = summarize(records)
    for path, digest in list(hashes.items()):
        if not Path(path).exists() or sha256(Path(path)) != digest:
            issues.append(f"Input artifact changed during analysis: {path}")
    conditions = condition_summaries(records)
    if issues:
        # A common-source/grid/integrity failure invalidates cohort inference,
        # even when individual files remain usable as descriptive evidence.
        for contrast in contrasts:
            contrast.update(status="unavailable", interval=None, cohort_issues=list(issues))
        for frontier in frontiers:
            frontier.update(smallest_tested_budget="unavailable", cohort_issues=list(issues))
            for budget in frontier["budgets"]:
                budget.update(interval=None, qualifies=None)
        for condition in conditions:
            condition.update(
                status="unavailable",
                means={},
                intervals={},
                interventions={},
                cohort_issues=list(issues),
            )
    passed = not issues and all(r["validation_passed"] for r in records.values())
    return {
        "schema_version": 1,
        "analysis": "pcp_visibility_budget_seed_analysis_v1",
        "checks_passed": passed,
        "status": "complete_for_review" if passed else "incomplete_or_invalid",
        "approval_decision": "not_issued",
        "issues": issues,
        "expected_policies": 40,
        "valid_policies": sum(r["validation_passed"] for r in records.values()),
        "training_seeds": list(TRAINING_SEEDS),
        "heldout_episode_seeds": EPISODE_SEEDS,
        "deadline": DEADLINE,
        "target": TARGET,
        "protocol_sha256": protocol_hash,
        "plan_sha256": plan_hash,
        "training_source_sha256": next(iter(sources)) if len(sources) == 1 else None,
        "analysis_source_sha256": source_fingerprint(ROOT),
        "artifact_hashes": hashes,
        "rows": list(records.values()),
        "contrasts": contrasts,
        "budget_frontiers": frontiers,
        "conditions": conditions,
        "primary_contrast": "broadcast_k3_minus_pcp_local_capacity/radius1",
        "interpretation": [
            (
                "Episodes are averaged within each policy; "
                "all five training seeds are required for an interval."
            ),
            (
                "Intervals are descriptive paired seed percentile bootstraps: "
                "10000 resamples, seed 73191."
            ),
            (
                "Failed, missing or invalid policies remain visible; "
                "affected contrasts have no reduced-seed CI."
            ),
            "First rewarded contact by step50 is not distinct capture or proof of coordination.",
            (
                "Smallest tested budget is conditional on this family, compute and seeds; "
                "K1/K2 are untested."
            ),
            "Severing/suppression/shuffling measure frozen-policy reliance and distribution shift.",
            (
                "No convergence, population superiority, CUDA, "
                "decentralized bandwidth bound or approval claim."
            ),
        ],
    }


def write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_figures(report, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels = dict(
        zip(MODELS, ("Identity", "LocalCapacity", "Broadcast K=0", "Broadcast K=3"), strict=True)
    )
    colors = dict(zip(MODELS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), strict=True))
    valid = [row for row in report["rows"] if row["validation_passed"]]
    state = (
        "COMPLETE EVIDENCE — review required"
        if report["checks_passed"]
        else f"INCOMPLETE EVIDENCE — {len(valid)}/40 valid policies"
    )

    def save(figure, name):
        figure.suptitle(f"PCP CPU · fixed 600k per policy\n{state}", fontsize=12)
        figure.text(
            0.5,
            0.030,
            "Dots/lines are trained policies; five seeds per condition. "
            "Descriptive seed uncertainty; no convergence claim.",
            ha="center",
            fontsize=8,
        )
        figure.text(
            0.5,
            0.008,
            f"Training source: {str(report['training_source_sha256'])[:12]} · "
            f"Protocol: {report['protocol_sha256'][:12]} · "
            f"Plan: {report['plan_sha256'][:12]} · full hashes in provenance.json",
            ha="center",
            fontsize=7,
        )
        figure.tight_layout(rect=(0, 0.065, 1, 0.93))
        for suffix in ("pdf", "svg", "png"):
            figure.savefig(out_dir / f"{name}.{suffix}", dpi=200)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, visibility in zip(axes, VISIBILITIES, strict=True):
        for model in MODELS:
            rows = [r for r in valid if r["model"] == model and r["visibility"] == visibility]
            for index, row in enumerate(rows):
                ax.plot(
                    EVALUATION_FRAMES,
                    row["performance"]["evaluation_returns"],
                    color=colors[model],
                    alpha=0.3,
                    linewidth=0.8,
                    label=f"{labels[model]} ({len(rows)}/5)"
                    if index == 0 and len(rows) < 5
                    else None,
                )
            if len(rows) == 5:
                values = np.asarray([r["performance"]["evaluation_returns"] for r in rows])
                ax.plot(
                    EVALUATION_FRAMES,
                    values.mean(0),
                    color=colors[model],
                    linewidth=1.8,
                    label=f"{labels[model]} (5/5)",
                )
        ax.set(title=visibility, xlabel="Collected training frames", xlim=(6000, 600000))
        ax.set_xticks([6000, *range(120000, 600001, 120000)])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("Online deterministic predator return")
    save(figure, "training_returns")

    panels = (
        ("contact_by_deadline", "Contact by step 50", "Probability"),
        ("contact_duration_steps", "Contact duration", "Steps per episode"),
        ("simultaneous_contact_steps", "Simultaneous contact", "Steps per episode"),
        ("predator_boundary_fraction", "Predator boundary occupancy", "Transition fraction"),
        ("prey_boundary_fraction", "Prey boundary occupancy", "Transition fraction"),
        ("action_saturation_fraction", "Predator action saturation", "Fraction |action| > .999"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    conditions = {(r["model"], r["visibility"]): r for r in report["conditions"]}
    slots = list(itertools.product(VISIBILITIES, MODELS))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        for x, (visibility, model) in enumerate(slots):
            rows = [r for r in valid if r["visibility"] == visibility and r["model"] == model]
            for row in rows:
                offset = (row["seed"] - 22) * 0.07
                ax.scatter(
                    x + offset, row["arms"]["live"][metric], s=18, color=colors[model], alpha=0.8
                )
            interval = conditions[(model, visibility)]["intervals"].get(metric)
            if interval:
                mean = interval["mean"]
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[mean - interval["ci95_low"]], [interval["ci95_high"] - mean]],
                    fmt="_",
                    color="black",
                    capsize=4,
                    markersize=13,
                )
        ax.set_xticks(range(8), [f"{v}\n{labels[m]}" for v, m in slots], rotation=55, ha="right")
        ax.set(title=title, ylabel=ylabel)
        if metric == "contact_by_deadline":
            ax.axhline(TARGET, color="black", linestyle="--", linewidth=0.8)
            ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.2)
    save(figure, "heldout_outcomes")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True)
    for ax, frontier in zip(axes, report["budget_frontiers"], strict=True):
        for budget in frontier["budgets"]:
            k = budget["sender_budget"]
            for row in budget["per_seed"]:
                ax.scatter(k + (row["seed"] - 22) * 0.035, row["value"], color="#0072B2", s=22)
            interval = budget["interval"]
            if interval:
                mean = interval["mean"]
                ax.errorbar(
                    k,
                    mean,
                    yerr=[[mean - interval["ci95_low"]], [interval["ci95_high"] - mean]],
                    fmt="_",
                    color="black",
                    capsize=5,
                    markersize=14,
                )
        ax.axhline(TARGET, color="black", linestyle="--", linewidth=0.8, label="Target = .80")
        ax.set(
            xticks=[0, 3],
            xlim=(-0.4, 3.4),
            ylim=(-0.03, 1.03),
            xlabel="Tested sender budget K",
            title=f"{frontier['visibility']} · smallest qualifying tested K: "
            f"{frontier['smallest_tested_budget']}",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out probability of contact by step 50")
    save(figure, "tested_budget_frontier")


def write_outputs(report, out_dir: Path):
    require(not out_dir.exists(), "Refusing to overwrite existing analysis output")
    out_dir.mkdir(parents=True)
    per_seed, episodes, curves = [], [], []
    for row in report["rows"]:
        identity = {key: row[key] for key in ("run_id", "model", "visibility", "seed", "status")}
        identity["issues"] = json.dumps(row["issues"])
        if not row["validation_passed"]:
            per_seed.append(identity)
            continue
        for arm, metrics in row["arms"].items():
            per_seed.append({**identity, "arm": arm, **metrics})
            for episode in row["episode_records"][arm]:
                episodes.append(
                    {
                        **identity,
                        "arm": arm,
                        "episode_id": episode["episode_id"],
                        "episode_seed": episode["seed"],
                        **episode["metrics"],
                        **{f"action_{k}": v for k, v in episode["action_health"].items()},
                        **{f"cost_{k}": v for k, v in episode["costs"].items()},
                    }
                )
        curves.extend(
            {**identity, "frames": frame, "return_mean": value}
            for frame, value in zip(
                EVALUATION_FRAMES, row["performance"]["evaluation_returns"], strict=True
            )
        )
    write_csv(out_dir / "per_seed.csv", per_seed)
    write_csv(out_dir / "episode_metrics.csv", episodes)
    write_csv(out_dir / "training_curves.csv", curves)
    render_figures(report, out_dir)
    serialized = {
        **report,
        "rows": [
            {k: v for k, v in row.items() if k != "episode_records"} for row in report["rows"]
        ],
    }
    (out_dir / "analysis_report.json").write_text(
        json.dumps(serialized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    provenance = {
        "schema_version": 1,
        "input_artifact_hashes": report["artifact_hashes"],
        "output_artifact_hashes": {
            p.name: sha256(p) for p in sorted(out_dir.iterdir()) if p.is_file()
        },
        "script_sha256": sha256(Path(__file__)),
        "analysis_source_sha256": report["analysis_source_sha256"],
        "training_source_sha256": report["training_source_sha256"],
        "episode_csv_scope": (
            "Complete validated policies only; every planned policy remains in per_seed.csv"
        ),
        "native_checkpoint_scope": (
            "Rehashed all five files against the CLI-validated native-header inventory"
        ),
        "no_training_or_policy_rollout": True,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluations-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/protocols/pcp_visibility_budget_cpu_v1.yaml",
    )
    parser.add_argument("--plan", type=Path, default=ROOT / "docs/PCP_VISIBILITY_BUDGET_PLAN.md")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error("Refusing to overwrite existing output; use a new analysis directory.")
    if args.out_dir.resolve().is_relative_to(args.evaluations_dir.resolve()) or (
        args.out_dir.resolve().is_relative_to(args.manifest.parent.resolve())
    ):
        parser.error("Analysis output must be outside input evidence and the managed suite.")
    report = analyze(
        args.manifest, args.evaluations_dir, protocol_path=args.protocol, plan_path=args.plan
    )
    write_outputs(report, args.out_dir)
    print(
        json.dumps(
            {
                "checks_passed": report["checks_passed"],
                "valid_policies": report["valid_policies"],
                "report": str(args.out_dir / "analysis_report.json"),
            }
        )
    )
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
