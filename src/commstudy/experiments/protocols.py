"""Versioned scientific protocols and evidence-bound launch authorization.

Planning never approves a protocol. A reviewed decision binds an exact protocol,
source tree, stage, and evidence files. Both the pre-submission CLI and managed
workers validate those bindings; a stale CSV cannot authorize a changed run.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from commstudy.experiments.config import (
    ExperimentSpec,
    experiment_spec_from_dict,
    scientific_config_sha256,
    resolved_experiment_dict,
)
from commstudy.experiments.provenance import source_fingerprint


PCP_TASK = "vmas_predator_capture_prey"
STAGES = {"confirmation", "comparison", "ablation"}
PROTOCOL_FIELDS = {
    "schema_version",
    "protocol_id",
    "status",
    "description",
    "base_spec",
    "methods",
    "factors",
    "runtime",
    "contracts",
    "selection",
    "evidence",
}
OPTIONAL_PROTOCOL_FIELDS = {"launch_scope"}
CONDITION_FIELDS = {"model", "ablation", "ablation_value"}
RUNTIME_FIELDS = {"device", "threads", "python", "packages"}
APPROVAL_FIELDS = {
    "schema_version",
    "protocol_id",
    "protocol_sha256",
    "source_sha256",
    "stage",
    "decision",
    "reviewer",
    "rationale",
    "evidence",
    "validated_factors",
}


class ProtocolGateError(ValueError):
    """An experimental launch has not met its declared protocol contract."""


def content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _exact_keys(value, allowed, label, required=()):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ValueError(
            f"{label}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}."
        )


def load_protocol(path: Path) -> dict[str, Any]:
    document = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    _exact_keys(document, PROTOCOL_FIELDS | OPTIONAL_PROTOCOL_FIELDS, "Protocol", PROTOCOL_FIELDS)
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["status"] not in {"candidate", "frozen"}
    ):
        raise ValueError("Unsupported protocol schema or status.")
    if not isinstance(document["protocol_id"], str) or not document["protocol_id"]:
        raise ValueError("Protocol ID must be nonempty.")
    base = experiment_spec_from_dict(document["base_spec"])
    if base.task != PCP_TASK:
        raise ValueError("This protocol gate currently supports PCP only.")
    _exact_keys(document["runtime"], RUNTIME_FIELDS, "Protocol runtime", RUNTIME_FIELDS)
    runtime = document["runtime"]
    if runtime["device"] not in {"cpu", "cuda"}:
        raise ValueError("Protocol runtime device must be cpu or cuda.")
    if type(runtime["threads"]) is not int or runtime["threads"] < 1:
        raise ValueError("Protocol threads must be a positive integer.")
    if not isinstance(runtime["packages"], Mapping) or not runtime["packages"]:
        raise ValueError("Protocol must pin runtime packages.")
    if not isinstance(runtime["python"], str) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in runtime["packages"].items()
    ):
        raise ValueError("Runtime Python/package versions must be explicit strings.")
    contract_fields = {
        "randomness_protocol",
        "observation_protocol",
        "critic_state_protocol",
        "evaluation_randomness_protocol",
        "channel_randomness_protocol",
        "training_health_protocol",
        "domain_metrics_protocol",
        "time_limit_protocol",
        "reward_protocol",
        "training_groups",
        "measured_groups",
        "prey_policy",
    }
    _exact_keys(document["contracts"], contract_fields, "Protocol contracts", contract_fields)
    for field in ("training_groups", "measured_groups"):
        values = document["contracts"][field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Protocol {field} must be a nonempty unique group list.")
    if document["contracts"]["measured_groups"] != base.task_config.get("return_groups"):
        raise ValueError("Protocol measured_groups must match the task's return_groups.")
    if document["contracts"]["reward_protocol"] != "simple_tag_repeated_contact_v1":
        raise ValueError("Unsupported PCP reward_protocol.")
    if document["contracts"]["prey_policy"] != "discarded_by_script_but_still_optimized":
        raise ValueError("Unsupported PCP prey_policy contract.")
    selection_fields = {
        "criterion",
        "budget_claim",
        "confirmation_seeds",
        "comparison_seeds",
        "ablation_seeds",
        "comparison_requires",
        "ablation_requires",
    }
    _exact_keys(document["selection"], selection_fields, "Protocol selection", selection_fields)
    for stage in STAGES:
        seeds = document["selection"][f"{stage}_seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise ValueError(f"Protocol {stage} seeds must be unique nonnegative integers.")
    if not isinstance(document["evidence"], list):
        raise ValueError("Protocol evidence references must be a list.")
    for item in document["evidence"]:
        _exact_keys(item, {"path", "role"}, "Protocol evidence reference", {"path", "role"})
    if not isinstance(document["methods"], Mapping) or not document["methods"]:
        raise ValueError("Protocol must declare complete model specifications.")
    for model, config in document["methods"].items():
        experiment_spec_from_dict({**document["base_spec"], "model": model, "model_config": config})
    if not isinstance(document["factors"], Mapping):
        raise ValueError("Protocol factors must be a mapping.")
    for name, factor in document["factors"].items():
        _exact_keys(
            factor,
            {"models", "values", "requires_evidence"},
            f"Factor {name}",
            {"models", "values"},
        )
        if (
            not isinstance(factor["models"], list)
            or not factor["models"]
            or any(not isinstance(model, str) for model in factor["models"])
            or len(set(factor["models"])) != len(factor["models"])
            or set(factor["models"]) - set(document["methods"])
        ):
            raise ValueError(f"Factor {name} names missing methods.")
        if not isinstance(factor["values"], Mapping) or not factor["values"]:
            raise ValueError(f"Factor {name} requires explicit value-to-patch mappings.")
        guarded = factor.get("requires_evidence", [])
        if (
            not isinstance(guarded, list)
            or any(not isinstance(value, str) for value in guarded)
            or len(set(guarded)) != len(guarded)
            or set(guarded) - {str(value) for value in factor["values"]}
        ):
            raise ValueError(f"Factor {name} requires_evidence must list declared factor values.")
        for value, by_model in factor["values"].items():
            if not isinstance(by_model, Mapping) or not by_model:
                raise ValueError(f"Factor {name}/{value} requires a nonempty patch.")
            if set(by_model) - set(factor["models"]):
                raise ValueError(f"Factor {name}/{value} contains an undeclared method.")
            if any(not isinstance(patch, Mapping) or not patch for patch in by_model.values()):
                raise ValueError(f"Factor {name}/{value} requires explicit patches per method.")
            if any(not isinstance(key, str) or key.split(".", 1)[0]
                   not in {"task_config", "model_config"}
                   for patch in by_model.values() for key in patch):
                raise ValueError("Protocol factors may change task_config/model_config only.")
    if "launch_scope" in document:
        scope = document["launch_scope"]
        _exact_keys(scope, STAGES, "Protocol launch_scope")
        if not scope:
            raise ValueError("Protocol launch_scope must authorize at least one explicit stage.")
        for stage, conditions in scope.items():
            if not isinstance(conditions, list) or not conditions:
                raise ValueError(f"launch_scope/{stage} must list explicit conditions.")
            seen = set()
            for condition in conditions:
                _exact_keys(condition, CONDITION_FIELDS, "Launch condition", CONDITION_FIELDS)
                if (
                    not isinstance(condition["model"], str)
                    or not isinstance(condition["ablation"], str)
                    or (condition["ablation_value"] is not None
                        and not isinstance(condition["ablation_value"], str))
                ):
                    raise ValueError(
                        "Launch condition labels must be strings (or a null main value)."
                    )
                identity = content_sha256(condition)
                if identity in seen:
                    raise ValueError(f"Duplicate launch condition in stage {stage}.")
                seen.add(identity)
                protocol_spec(document, seed=base.seed, **condition)
                validate_protocol_condition(document, stage=stage,
                                            seed=document["selection"][f"{stage}_seeds"][0],
                                            **condition)
    return document


def protocol_spec(
    protocol: Mapping[str, Any],
    *,
    model: str,
    seed: int,
    ablation: str = "main",
    ablation_value: str | None = None,
) -> ExperimentSpec:
    """Resolve only a declared method, seed, and exact experimental factor."""
    if model not in protocol["methods"]:
        raise ProtocolGateError(f"Method {model!r} is not declared by this protocol.")
    raw = copy.deepcopy(protocol["base_spec"])
    raw.update(model=model, seed=seed, model_config=copy.deepcopy(protocol["methods"][model]))
    if ablation != "main":
        factor = protocol["factors"].get(ablation)
        if factor is None or model not in factor["models"]:
            raise ProtocolGateError(f"Undeclared factor {ablation!r} for method {model!r}.")
        patches = {str(key): value for key, value in factor["values"].items()}
        if str(ablation_value) not in patches:
            raise ProtocolGateError(f"Undeclared factor value {ablation}/{ablation_value}.")
        if model not in patches[str(ablation_value)]:
            raise ProtocolGateError(f"Factor {ablation}/{ablation_value} is undefined for {model}.")
        config = OmegaConf.create(raw)
        for key, value in patches[str(ablation_value)][model].items():
            # Factor schemas are validated through the resulting strict spec.
            OmegaConf.update(config, key, value, merge=False)
        raw = OmegaConf.to_container(config, resolve=True)
    elif ablation_value is not None:
        raise ProtocolGateError("Main rows cannot carry an undeclared ablation value.")
    return experiment_spec_from_dict(raw)


def validate_protocol_condition(
    protocol: Mapping[str, Any], *, stage: str, model: str, seed: int,
    ablation: str = "main", ablation_value: str | None = None,
) -> None:
    """The same exact condition gate is used by planning and worker launches.

    An explicit scope closes every omitted stage and permits comparison factors
    without relabeling a prespecified comparison as a later ablation.
    """
    if stage not in STAGES:
        raise ProtocolGateError(f"Unknown launch stage {stage!r}.")
    if type(seed) is not int or seed not in protocol["selection"][f"{stage}_seeds"]:
        raise ProtocolGateError("Seed is not declared for this protocol launch stage.")
    if stage == "confirmation" and (model != "pcp_comm_identity" or ablation != "main"):
        raise ProtocolGateError("Candidate confirmation is Identity-only without ablations.")
    if "launch_scope" in protocol:
        condition = {"model": model, "ablation": ablation, "ablation_value": ablation_value}
        if condition not in protocol["launch_scope"].get(stage, []):
            raise ProtocolGateError("Condition is outside the protocol's exact launch_scope.")
        if stage == "ablation" and ablation == "main":
            raise ProtocolGateError("Ablation stage requires a declared factor.")
    elif (stage == "ablation") != (ablation != "main"):
        raise ProtocolGateError("Launch stage conflicts with the declared factor.")


def runtime_mismatches(runtime: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    mismatches = {}
    if platform.python_version() != runtime["python"]:
        mismatches["python"] = [runtime["python"], platform.python_version()]
    for name, wanted in runtime["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if wanted != actual:
            mismatches[name] = [wanted, actual]
    if torch.get_num_threads() != runtime["threads"]:
        mismatches["torch_num_threads"] = [runtime["threads"], torch.get_num_threads()]
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != str(runtime["threads"]):
            mismatches[name] = [str(runtime["threads"]), os.environ.get(name)]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        mismatches["cuda_available"] = [True, False]
    return mismatches


def _bound_path(reference: Mapping[str, Any], root: Path) -> Path:
    _exact_keys(reference, {"path", "sha256"}, "Bound evidence", {"path", "sha256"})
    if not isinstance(reference["path"], str) or not reference["path"]:
        raise ProtocolGateError("Bound evidence requires a file path.")
    path = Path(reference["path"])
    path = path if path.is_absolute() else root / path
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ProtocolGateError(f"Missing or changed bound evidence: {path}")
    return path


def _bound_json(reference: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    value = json.loads(_bound_path(reference, root).read_text())
    if not isinstance(value, Mapping):
        raise ProtocolGateError("Bound evidence JSON must be a mapping.")
    return value


def _bridge_metrics(path: Path) -> dict[tuple, str]:
    values = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = {
            "timestamp", "frames", "iteration", "phase", "group", "metric", "sample", "value",
        }
        if set(reader.fieldnames or []) != columns or len(reader.fieldnames) != len(columns):
            raise ProtocolGateError("Identity bridge metrics have an unsupported CSV schema.")
        for row in reader:
            frame, iteration = int(row["frames"]), int(row["iteration"])
            value = float(row["value"])
            if not 0 <= frame <= 6000 or iteration < 0 or not math.isfinite(value):
                raise ProtocolGateError(
                    "Identity bridge must contain finite bounded 6000-frame metrics."
                )
            if row["metric"] == "wall_time_seconds":
                continue
            key = (frame, iteration, row["phase"], row["group"], row["metric"], row["sample"])
            if key in values:
                raise ProtocolGateError("Duplicate Identity bridge metric.")
            values[key] = value.hex()
    if not values or max(key[0] for key in values) != 6000:
        raise ProtocolGateError("Identity bridge metrics must reach exactly 6000 frames.")
    return values


def _validate_source_bridge(reference, *, old_source, new_source, repo_root):
    bridge = _bound_json(reference, repo_root)
    fields = {"schema_version", "approved", "reviewer", "rationale",
              "previous_source_sha256", "source_sha256", "identity_equivalence"}
    _exact_keys(bridge, fields, "Source bridge", fields)
    if (type(bridge["schema_version"]) is not int or bridge["schema_version"] != 1
            or bridge["approved"] is not True
            or bridge["previous_source_sha256"] != old_source
            or bridge["source_sha256"] != new_source
            or any(not isinstance(bridge[key], str) or not bridge[key].strip()
                   for key in ("reviewer", "rationale"))):
        raise ProtocolGateError("Source bridge is not approved for the exact source pair.")
    report = _bound_json(bridge["identity_equivalence"], repo_root)
    fields = {"schema_version", "checks_passed", "previous_source_sha256", "source_sha256",
              "model", "frames", "seed", "compared_scalar_count", "previous_metrics",
              "current_metrics", "previous_metadata", "current_metadata"}
    _exact_keys(report, fields, "Identity source equivalence", fields)
    if (type(report["schema_version"]) is not int or report["schema_version"] != 1
            or report["checks_passed"] is not True
            or report["previous_source_sha256"] != old_source
            or report["source_sha256"] != new_source
            or report["model"] != "pcp_comm_identity"
            or type(report["frames"]) is not int or report["frames"] != 6000
            or type(report["seed"]) is not int or report["seed"] < 0
            or type(report["compared_scalar_count"]) is not int):
        raise ProtocolGateError("Source bridge requires bounded matched-seed Identity equivalence.")
    compared, metadata_rows = [], []
    for label, source in (("previous", old_source), ("current", new_source)):
        metadata_path = _bound_path(report[f"{label}_metadata"], repo_root)
        metadata = _bound_json(report[f"{label}_metadata"], repo_root)
        metrics_path = _bound_path(report[f"{label}_metrics"], repo_root)
        if (metadata.get("source_sha256") != source or metadata.get("model") != report["model"]
                or metadata.get("task") != PCP_TASK or metadata.get("algorithm") != "mappo"
                or type(metadata.get("seed")) is not int
                or metadata.get("seed") != report["seed"]
                or metadata_path.parent.resolve() != metrics_path.parent.resolve()):
            raise ProtocolGateError(
                "Identity bridge run metadata does not match its source/seed/model."
            )
        metadata_rows.append(metadata)
        compared.append(_bridge_metrics(metrics_path))
    scientific_hash = metadata_rows[0].get("scientific_config_sha256")
    runtime_fields = {"platform", "device", "torch_num_threads", "cuda_available",
                      "cuda_device_name", "sampling_device", "train_device", "buffer_device"}
    if (not isinstance(scientific_hash, str) or len(scientific_hash) != 64
            or metadata_rows[1].get("scientific_config_sha256") != scientific_hash
            or not isinstance(metadata_rows[0].get("runtime"), Mapping)
            or not isinstance(metadata_rows[1].get("runtime"), Mapping)
            or {key: metadata_rows[0]["runtime"].get(key) for key in runtime_fields}
            != {key: metadata_rows[1]["runtime"].get(key) for key in runtime_fields}
            or not isinstance(metadata_rows[0].get("versions"), Mapping)
            or metadata_rows[0]["versions"] != metadata_rows[1].get("versions")):
        raise ProtocolGateError("Identity bridge must preserve the scientific spec and runtime.")
    if compared[0] != compared[1] or len(compared[0]) != report["compared_scalar_count"]:
        raise ProtocolGateError(
            "Identity bridge numeric metrics differ or the compared count is false."
        )


def _validate_promotion_certificate(reference, *, protocol, stage, repo_root):
    """A kind label alone cannot turn a failed or unrelated packet into promotion."""
    certificate = _bound_json(reference, repo_root)
    fields = {"schema_version", "decision", "authorized_protocol_sha256", "authorized_stage",
              "confirmed_protocol", "review", "report", "source_bridge"}
    _exact_keys(certificate, fields, "Promotion certificate", fields)
    if (type(certificate["schema_version"]) is not int or certificate["schema_version"] != 1
            or certificate["decision"] != "confirmed"
            or certificate["authorized_protocol_sha256"] != content_sha256(protocol)
            or certificate["authorized_stage"] != stage):
        raise ProtocolGateError(
            "Promotion certificate is not confirmed for this protocol and stage."
        )
    confirmed = load_protocol(_bound_path(certificate["confirmed_protocol"], repo_root))
    # Promotion carries a confirmed training/runtime baseline into a controlled
    # comparison. Method and task interventions remain explicit factor choices.
    for field in ("algorithm", "task", "algorithm_config", "task_config", "model_config",
                  "critic_model", "experiment"):
        target_value = copy.deepcopy(protocol["base_spec"][field])
        confirmed_value = copy.deepcopy(confirmed["base_spec"][field])
        if field == "experiment":
            target_value.pop("save_folder", None)
            confirmed_value.pop("save_folder", None)
        if target_value != confirmed_value:
            raise ProtocolGateError(f"Promotion changes the confirmed baseline {field}.")
    for field in ("runtime", "contracts"):
        if protocol[field] != confirmed[field]:
            raise ProtocolGateError(f"Promotion changes the confirmed baseline {field}.")
    review = _bound_json(certificate["review"], repo_root)
    report = _bound_json(certificate["report"], repo_root)
    prerequisite = "confirmation" if stage == "comparison" else "comparison"
    report_hash_key = f"{prerequisite}_report_sha256"
    source = report.get("source_sha256")
    expected = {"protocol_id": confirmed["protocol_id"],
                "protocol_sha256": content_sha256(confirmed), "source_sha256": source}
    if (not isinstance(source, str) or len(source) != 64
            or any(letter not in "0123456789abcdef" for letter in source)
            or any(report.get(key) != value or review.get(key) != value
                   for key, value in expected.items())
            or review.get("overall_verdict") != "CONFIRMED"
            or review.get("completion_gate_passed") is not True
            or review.get("stage") != f"{prerequisite}_review_only"
            or review.get(report_hash_key) != certificate["report"]["sha256"]
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or report.get("checks_passed") is not True or report.get("issues") != []):
        raise ProtocolGateError(
            "Promotion requires a completed, confirmed review and matching valid report."
        )
    seeds = confirmed["selection"][f"{prerequisite}_seeds"]
    runs = report.get("runs")
    if (report.get("expected_seeds") != seeds or not isinstance(runs, list) or not runs
            or any(not isinstance(row, Mapping) or row.get("validation_passed") is not True
                   or type(row.get("seed")) is not int for row in runs)):
        raise ProtocolGateError(
            "Promotion report does not contain the complete validated seed set."
        )
    if prerequisite == "confirmation":
        complete = sorted(row["seed"] for row in runs) == sorted(seeds)
    else:
        conditions = confirmed.get("launch_scope", {}).get("comparison", [
            {"model": model, "ablation": "main", "ablation_value": None}
            for model in confirmed["methods"]
        ])
        expected_rows = {content_sha256({**condition, "seed": seed})
                         for condition in conditions for seed in seeds}
        actual_rows = [content_sha256({key: row.get(key) for key in CONDITION_FIELDS | {"seed"}})
                       for row in runs]
        complete = len(actual_rows) == len(expected_rows) and set(actual_rows) == expected_rows
    if not complete:
        raise ProtocolGateError("Promotion report does not contain the complete validated row set.")
    evidence = review.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ProtocolGateError("Promotion review must retain its hashed evidence packet.")
    for item in evidence.values():
        _bound_path(item, repo_root)
    current_source = source_fingerprint(repo_root)
    if source != current_source:
        if certificate["source_bridge"] is None:
            raise ProtocolGateError(
                "Promotion across changed source requires an approved Identity bridge."
            )
        _validate_source_bridge(certificate["source_bridge"], old_source=source,
                                new_source=current_source, repo_root=repo_root)
    elif certificate["source_bridge"] is not None:
        _validate_source_bridge(certificate["source_bridge"], old_source=source,
                                new_source=current_source, repo_root=repo_root)


def validate_approval(
    protocol: Mapping[str, Any],
    approval_path: Path,
    *,
    stage: str,
    repo_root: Path,
    ablation: str = "main",
    ablation_value: str | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ProtocolGateError(f"Unknown launch stage {stage!r}.")
    if not approval_path.is_file():
        raise ProtocolGateError(f"Protocol approval is missing: {approval_path}")
    approval = json.loads(approval_path.read_text())
    _exact_keys(approval, APPROVAL_FIELDS, "Protocol approval", APPROVAL_FIELDS)
    expected = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": content_sha256(protocol),
        "source_sha256": source_fingerprint(repo_root),
        "stage": stage,
        "decision": "approved",
    }
    if type(approval["schema_version"]) is not int or any(
        approval[key] != value for key, value in expected.items()
    ):
        raise ProtocolGateError("Approval does not match this protocol, source, and launch stage.")
    if any(
        not isinstance(approval[key], str) or not approval[key].strip()
        for key in ("reviewer", "rationale")
    ):
        raise ProtocolGateError("Approval must record its reviewer and rationale.")
    if not isinstance(approval["validated_factors"], list) or any(
        not isinstance(value, str) for value in approval["validated_factors"]
    ):
        raise ProtocolGateError("validated_factors must be a list of factor IDs.")
    evidence = approval["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ProtocolGateError("Approval requires recorded evidence.")
    for item in evidence:
        _exact_keys(
            item, {"path", "sha256", "kind"}, "Approval evidence", {"path", "sha256", "kind"}
        )
        path = Path(item["path"])
        path = path if path.is_absolute() else repo_root / path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ProtocolGateError(f"Missing or changed approval evidence: {path}")
    kinds = {item["kind"] for item in evidence}
    required = {
        "confirmation": "candidate_review",
        "comparison": "corrected_protocol_confirmation",
        "ablation": "main_comparison_review",
    }[stage]
    if required not in kinds:
        raise ProtocolGateError(f"Stage {stage} requires {required} evidence.")
    if stage != "confirmation" and protocol["status"] != "frozen":
        raise ProtocolGateError("Comparison and ablation require a frozen protocol.")
    if stage != "confirmation":
        certificates = [item for item in evidence if item["kind"] == required]
        if len(certificates) != 1:
            raise ProtocolGateError("A launch stage requires exactly one promotion certificate.")
        _validate_promotion_certificate(
            {key: certificates[0][key] for key in ("path", "sha256")},
            protocol=protocol, stage=stage, repo_root=repo_root,
        )
    if ablation != "main":
        factor = protocol["factors"][ablation]
        required_factor = f"{ablation}/{ablation_value}"
        guarded = {str(value) for value in factor.get("requires_evidence", [])}
        if str(ablation_value) in guarded and required_factor not in approval["validated_factors"]:
            raise ProtocolGateError(f"Factor {required_factor} needs separate validation evidence.")
        if str(ablation_value) in guarded and f"factor:{required_factor}" not in kinds:
            raise ProtocolGateError(
                f"Factor {required_factor} needs hashed factor-specific evidence."
            )
    return approval


def validate_protocol_launch(
    spec: ExperimentSpec,
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    check_runtime: bool = True,
) -> dict[str, Any]:
    """Validate a manifest's immutable protocol binding before managed training."""
    required = {
        "path",
        "sha256",
        "source_sha256",
        "approval",
        "stage",
        "model",
        "seed",
        "ablation",
        "ablation_value",
    }
    _exact_keys(binding, required, "Protocol binding", required)
    path = Path(binding["path"])
    path = path if path.is_absolute() else repo_root / path
    protocol = load_protocol(path)
    stage = binding["stage"]
    validate_protocol_condition(protocol, stage=stage, model=binding["model"],
                                seed=binding["seed"], ablation=binding["ablation"],
                                ablation_value=binding["ablation_value"])
    if content_sha256(protocol) != binding["sha256"]:
        raise ProtocolGateError("Protocol content changed since this manifest was generated.")
    if source_fingerprint(repo_root) != binding["source_sha256"]:
        raise ProtocolGateError("Research source changed since this manifest was generated.")
    expected = protocol_spec(
        protocol,
        model=binding["model"],
        seed=binding["seed"],
        ablation=binding["ablation"],
        ablation_value=binding["ablation_value"],
    )
    if scientific_config_sha256(spec) != scientific_config_sha256(expected):
        raise ProtocolGateError("Resolved row differs from its declared protocol/factor/seed.")
    actual_full, expected_full = resolved_experiment_dict(spec), resolved_experiment_dict(expected)
    for raw in (actual_full, expected_full):
        # Output placement is portable. Logging/evaluation/checkpoint/restore
        # settings are frozen because they affect execution or required evidence.
        raw["experiment"].pop("save_folder", None)
    if actual_full != expected_full:
        raise ProtocolGateError(
            "Full resolved protocol differs (logging/checkpoints/restore/runtime)."
        )
    for key in ("sampling_device", "train_device", "buffer_device"):
        if str(spec.experiment.get(key, "cpu")).split(":")[0] != protocol["runtime"]["device"]:
            raise ProtocolGateError(f"Row {key} violates the protocol runtime device.")
    approval_path = Path(binding["approval"])
    if not approval_path.is_absolute():
        approval_path = repo_root / approval_path
    approval = validate_approval(
        protocol,
        approval_path,
        stage=binding["stage"],
        repo_root=repo_root,
        ablation=binding["ablation"],
        ablation_value=binding["ablation_value"],
    )
    if check_runtime:
        mismatches = runtime_mismatches(protocol["runtime"])
        if mismatches:
            raise ProtocolGateError(f"Runtime does not match the approved protocol: {mismatches}")
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": binding["sha256"],
        "source_sha256": binding["source_sha256"],
        "stage": binding["stage"],
        "factors": {"model": binding["model"], binding["ablation"]: binding["ablation_value"]},
        "approval_sha256": content_sha256(approval),
    }


def expected_model_contract(spec: ExperimentSpec) -> dict[str, Any]:
    """Build the declared model on CPU and fingerprint actual classes/parameter shapes.

    Called once per unique architecture during manifest generation. No rollout,
    optimization, or scientific training is performed.
    """
    import tempfile
    from commstudy.experiments.runner import build_experiment
    from commstudy.utils.rng import preserve_rng_state

    with tempfile.TemporaryDirectory(prefix="commstudy-contract-") as temporary:
        inspection = dataclasses.replace(
            spec,
            experiment={
                **spec.experiment,
                "train_device": "cpu",
                "sampling_device": "cpu",
                "buffer_device": "cpu",
                "save_folder": temporary,
                "evaluation_episodes": 1,
                "on_policy_n_envs_per_worker": 1,
                "loggers": [],
                "create_json": False,
            },
        )
        with preserve_rng_state():
            experiment = build_experiment(inspection)
            try:
                return actual_model_contract(experiment)
            finally:
                experiment.close()


def actual_model_contract(experiment: Any) -> dict[str, Any]:
    from commstudy.experiments.bookkeeping import parameter_counts, task_runtime_contract

    def tree(module):
        return {
            "classes": sorted(
                {type(item).__module__ + "." + type(item).__qualname__ for item in module.modules()}
            ),
            "parameters": {key: list(value.shape) for key, value in module.named_parameters()},
        }

    return {
        "training_groups": sorted(experiment.train_group_map),
        "groups": {group: tree(policy) for group, policy in experiment.group_policies.items()},
        "critic": {
            group: tree(loss.critic_network)
            for group, loss in experiment.losses.items()
            if hasattr(loss, "critic_network")
        },
        "parameter_counts": parameter_counts(experiment),
        "task_contract": task_runtime_contract(experiment),
    }


def validate_built_contract(protocol, contract):
    """Check declared semantic versions and training groups against real construction."""
    declared = protocol["contracts"]
    actual = contract["task_contract"]
    aliases = {"critic_state_protocol": actual.get("critic_state", {}).get("protocol")}
    for key in (
        "randomness_protocol",
        "observation_protocol",
        "critic_state_protocol",
        "evaluation_randomness_protocol",
        "channel_randomness_protocol",
        "training_health_protocol",
        "domain_metrics_protocol",
        "time_limit_protocol",
    ):
        value = aliases[key] if key in aliases else actual.get(key)
        if declared[key] != value:
            raise ProtocolGateError(f"Declared {key} differs from the built task/model contract.")
    if sorted(declared["training_groups"]) != contract["training_groups"]:
        raise ProtocolGateError("Declared training groups differ from the built experiment.")
