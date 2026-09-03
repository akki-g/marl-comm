from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnalysisOutputs:
    per_run_csv: Path
    summary_csv: Path
    failed_runs_csv: Path
    plot_files: tuple[Path, ...]
    paired_comparisons_csv: Path | None = None

    @property
    def comparisons_csv(self) -> Path | None:
        """Backward-friendly short alias for the paired-comparison artifact."""

        return self.paired_comparisons_csv


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _evaluation_curve(rows: Sequence[Mapping[str, str]]) -> list[tuple[int, float]]:
    by_frame: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("phase") == "evaluation"
            and row.get("metric") == "return_mean"
            # The study figure is written without a group; per-group returns are
            # recorded alongside it and must not be averaged into the curve.
            and not row.get("group")
            and not row.get("sample")
        ):
            value = float(row["value"])
            if math.isfinite(value):
                by_frame[int(row["frames"])].append(value)
    return sorted((frame, float(np.mean(values))) for frame, values in by_frame.items())


def _auc(curve: Sequence[tuple[int, float]]) -> tuple[float | None, float | None]:
    if not curve:
        return None, None
    if len(curve) == 1:
        return 0.0, curve[0][1]
    x = np.asarray([point[0] for point in curve], dtype=float)
    y = np.asarray([point[1] for point in curve], dtype=float)
    value = float(np.trapezoid(y, x))
    span = float(x[-1] - x[0])
    return value, value / span if span else float(y[-1])


def compute_run_result(run_dir: Path) -> dict[str, Any]:
    metadata = _read_json(run_dir / "metadata.json")
    rows = _metric_rows(run_dir / "metrics.csv")
    curve = _evaluation_curve(rows)
    final_count = max(1, math.ceil(len(curve) * 0.10)) if curve else 0
    final_return = float(np.mean([value for _, value in curve[-final_count:]])) if curve else None
    raw_auc, normalized_auc = _auc(curve)

    comm_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "evaluation" and row.get("metric", "").startswith("comm_"):
            value = float(row["value"])
            if math.isfinite(value):
                comm_values[row["metric"]].append(value)

    result: dict[str, Any] = {
        "run_id": metadata["run_id"],
        "suite_id": metadata["suite_id"],
        "status": metadata.get("status", "completed"),
        "task": metadata["task"],
        "algorithm": metadata["algorithm"],
        "model": metadata["model"],
        "seed": int(metadata["seed"]),
        "ablation": metadata.get("ablation") or "main",
        "ablation_value": (
            None
            if metadata.get("ablation_value") in {None, ""}
            else str(metadata["ablation_value"])
        ),
        "attempt": int(metadata.get("attempt") or 0),
        "retry_of": metadata.get("retry_of"),
        "evaluation_points": len(curve),
        "final_window_points": final_count,
        "mean_final_return": final_return,
        "return_auc": raw_auc,
        "normalized_return_auc": normalized_auc,
        "training_time_seconds": metadata.get("benchmarl_total_time_seconds"),
    }
    parameters = metadata.get("parameters") or {}
    result.update(
        {
            "actor_params": parameters.get("actor_total"),
            "comm_params": parameters.get("communication_total"),
            "critic_params": parameters.get("critic_total"),
        }
    )
    result.update(
        {metric: float(np.mean(values)) for metric, values in comm_values.items() if values}
    )
    result.update(_saliency_fields(run_dir))
    return result


#: Saliency fields aggregated across seeds like any other numeric run metric.
SALIENCY_METRICS = (
    "saliency_return_with_comm",
    "saliency_return_without_comm",
    "saliency_return_delta",
    "saliency_return_delta_fraction",
    "saliency_action_shift_mean",
    "saliency_policy_kl_mean",
)


def _saliency_fields(run_dir: Path) -> dict[str, float]:
    """Read the optional per-run communication-saliency measurement.

    Saliency is produced by a separate post-hoc pass over finished
    checkpoints, so it may legitimately be absent. Missing is treated as "not
    measured" and simply omitted, never as zero.
    """

    path = run_dir / "saliency.json"
    if not path.exists():
        return {}
    document = _read_json(path)
    fields: dict[str, float] = {}
    for metric in SALIENCY_METRICS:
        value = document.get(metric)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            fields[metric] = numeric
    return fields


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    if samples < 1:
        raise ValueError("samples must be at least one")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _group_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        result["task"],
        result["algorithm"],
        result["model"],
        result["ablation"],
        result.get("ablation_value"),
    )


def _comparison_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        result["task"],
        result["algorithm"],
        result["ablation"],
        result.get("ablation_value"),
    )


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalized_ablation_value(value: Any) -> str | None:
    return None if value in {None, ""} else str(value)


def _status_from_run_dir(run_dir: Path, metadata: Mapping[str, Any]) -> str:
    status_path = run_dir / "status.json"
    if status_path.exists():
        status_document = _read_json(status_path)
        status = status_document.get("status")
        if status:
            return str(status)
    return str(metadata.get("status") or "unknown")


def _manifest_rows(suite_dir: Path) -> list[dict[str, Any]]:
    path = suite_dir / "manifest.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    normalized: list[dict[str, Any]] = []
    for row in rows:
        run_id = row.get("run_id")
        if not run_id:
            continue
        normalized.append(
            {
                "run_id": run_id,
                "suite_id": row.get("suite_id") or suite_dir.name,
                "task": row.get("task"),
                "algorithm": row.get("algorithm"),
                "model": row.get("model"),
                "seed": _as_int(row.get("seed")),
                "ablation": row.get("ablation") or "main",
                "ablation_value": _normalized_ablation_value(row.get("ablation_value")),
                "status": row.get("status") or "pending",
                "attempt": _as_int(row.get("attempt")),
                "retry_of": row.get("retry_of") or None,
                "max_n_frames": (
                    _as_int(row.get("max_n_frames"))
                    if row.get("max_n_frames") not in {None, ""}
                    else None
                ),
                "expected_in_manifest": True,
                "run_dir_present": False,
                "metrics_present": False,
                "analysis_ready": False,
            }
        )
    return normalized


def _inventory_rows(suite_dir: Path) -> list[dict[str, Any]]:
    """Join the planned manifest with durable run state and computed metrics."""

    manifest = _manifest_rows(suite_dir)
    by_run_id = {str(row["run_id"]): dict(row) for row in manifest}
    ordered_ids = [str(row["run_id"]) for row in manifest]

    for run_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _read_json(metadata_path)
        run_id = str(metadata.get("run_id") or run_dir.name)
        if run_id not in by_run_id:
            ordered_ids.append(run_id)
        record = by_run_id.setdefault(
            run_id,
            {
                "run_id": run_id,
                "suite_id": metadata.get("suite_id") or suite_dir.name,
                "expected_in_manifest": False,
            },
        )
        record.update(
            {
                "run_id": run_id,
                "suite_id": metadata.get("suite_id") or record.get("suite_id"),
                "task": metadata.get("task") or record.get("task"),
                "algorithm": metadata.get("algorithm") or record.get("algorithm"),
                "model": metadata.get("model") or record.get("model"),
                "seed": _as_int(metadata.get("seed"), default=_as_int(record.get("seed"))),
                "ablation": metadata.get("ablation") or record.get("ablation") or "main",
                "ablation_value": _normalized_ablation_value(
                    metadata.get("ablation_value", record.get("ablation_value"))
                ),
                "status": _status_from_run_dir(run_dir, metadata),
                "attempt": _as_int(metadata.get("attempt"), default=_as_int(record.get("attempt"))),
                "retry_of": metadata.get("retry_of") or record.get("retry_of"),
                "run_dir_present": True,
                "metrics_present": (run_dir / "metrics.csv").exists(),
                "analysis_ready": False,
                "error": metadata.get("error"),
            }
        )
        if record["status"] == "completed":
            record.update(compute_run_result(run_dir))
            # These inventory fields are intentionally not supplied by
            # compute_run_result and must survive the merge above.
            record["status"] = "completed"
            record["expected_in_manifest"] = bool(record.get("expected_in_manifest"))
            record["run_dir_present"] = True
            record["metrics_present"] = (run_dir / "metrics.csv").exists()
            record["analysis_ready"] = True

    return [by_run_id[run_id] for run_id in ordered_ids]


def _completed_seed_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    """Choose at most one completed attempt per statistical seed."""

    selected: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status") != "completed" or not row.get("analysis_ready"):
            continue
        seed = int(row["seed"])
        previous = selected.get(seed)
        candidate_key = (_as_int(row.get("attempt")), str(row.get("run_id", "")))
        previous_key = (
            (_as_int(previous.get("attempt")), str(previous.get("run_id", "")))
            if previous is not None
            else (-1, "")
        )
        if candidate_key >= previous_key:
            selected[seed] = row
    return selected


def _metric_summary(
    values: Sequence[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None, None, None
    mean, low, high = bootstrap_mean_ci(
        finite,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, std, low, high


def aggregate_suite(
    suite_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory = _inventory_rows(suite_dir)
    failed: list[dict[str, Any]] = []
    for result in inventory:
        if result.get("analysis_ready"):
            continue
        error = result.get("error") or {}
        error_type = error.get("type")
        error_message = error.get("message")
        if result.get("status") == "completed" and not result.get("run_dir_present"):
            error_type = "MissingRunArtifacts"
            error_message = "Manifest marks run completed, but its run directory is absent."
        failed.append(
            {
                "run_id": result.get("run_id"),
                "suite_id": result.get("suite_id"),
                "status": result.get("status", "unknown"),
                "task": result.get("task"),
                "algorithm": result.get("algorithm"),
                "model": result.get("model"),
                "seed": result.get("seed"),
                "ablation": result.get("ablation"),
                "ablation_value": result.get("ablation_value"),
                "attempt": result.get("attempt", 0),
                "retry_of": result.get("retry_of"),
                "run_dir_present": result.get("run_dir_present", False),
                "error_type": error_type,
                "error_message": error_message,
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in inventory:
        required = ("task", "algorithm", "model", "seed")
        if all(result.get(key) is not None for key in required):
            grouped[_group_key(result)].append(result)

    comparison_expected_seeds: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for result in inventory:
        if result.get("expected_in_manifest") and result.get("task") is not None:
            comparison_expected_seeds[_comparison_key(result)].add(int(result["seed"]))
    if not comparison_expected_seeds:
        for result in inventory:
            if result.get("task") is not None:
                comparison_expected_seeds[_comparison_key(result)].add(int(result["seed"]))

    summaries = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        completed_by_seed = _completed_seed_rows(group_rows)
        results = list(completed_by_seed.values())
        final_values = [
            float(result["mean_final_return"])
            for result in results
            if result["mean_final_return"] is not None
        ]
        auc_values = [
            float(result["normalized_return_auc"])
            for result in results
            if result["normalized_return_auc"] is not None
        ]
        mean_final, std_final, ci_low, ci_high = _metric_summary(
            final_values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        mean_auc, std_auc, auc_ci_low, auc_ci_high = _metric_summary(
            auc_values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        present_seeds = {int(result["seed"]) for result in results}
        own_expected_seeds = {
            int(result["seed"]) for result in group_rows if result.get("expected_in_manifest")
        }
        if not own_expected_seeds:
            own_expected_seeds = {int(result["seed"]) for result in group_rows}
        expected_seeds = comparison_expected_seeds[(key[0], key[1], key[3], key[4])]
        status_counts: dict[str, int] = defaultdict(int)
        for result in group_rows:
            status_counts[str(result.get("status") or "unknown")] += 1
        summary: dict[str, Any] = {
            "task": key[0],
            "algorithm": key[1],
            "model": key[2],
            "ablation": key[3],
            "ablation_value": key[4],
            "n_seeds": len(present_seeds),
            "n_expected_seeds": len(own_expected_seeds),
            "n_completed_seeds": len(present_seeds),
            "n_expected_runs": sum(
                bool(result.get("expected_in_manifest")) for result in group_rows
            )
            or len(group_rows),
            "n_completed_runs": sum(bool(result.get("analysis_ready")) for result in group_rows),
            "n_unavailable_runs": sum(
                not bool(result.get("analysis_ready")) for result in group_rows
            ),
            "n_pending_runs": status_counts.get("pending", 0),
            "n_running_runs": status_counts.get("running", 0),
            "n_failed_runs": status_counts.get("failed", 0),
            "expected_seeds": ";".join(str(seed) for seed in sorted(own_expected_seeds)),
            "seeds": ";".join(str(seed) for seed in sorted(present_seeds)),
            "missing_expected_seeds": ";".join(
                str(seed) for seed in sorted(own_expected_seeds - present_seeds)
            ),
            "missing_matched_seeds": ";".join(
                str(seed) for seed in sorted(expected_seeds - present_seeds)
            ),
            "mean_final_return": mean_final,
            "std_final_return": std_final,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "mean_normalized_auc": mean_auc,
            "std_normalized_auc": std_auc,
            "normalized_auc_ci95_low": auc_ci_low,
            "normalized_auc_ci95_high": auc_ci_high,
            "mean_training_time": float(
                np.mean(
                    [
                        result["training_time_seconds"]
                        for result in results
                        if result["training_time_seconds"] is not None
                    ]
                )
            )
            if any(result["training_time_seconds"] is not None for result in results)
            else None,
            "actor_params": results[0].get("actor_params") if results else None,
            "comm_params": results[0].get("comm_params") if results else None,
            "critic_params": results[0].get("critic_params") if results else None,
        }
        aggregated_keys = sorted(
            {
                key
                for result in results
                for key in result
                if key.startswith("comm_") or key.startswith("saliency_")
            }
        )
        for aggregated_key in aggregated_keys:
            values = [
                float(result[aggregated_key]) for result in results if aggregated_key in result
            ]
            mean, std, low, high = _metric_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            summary[f"mean_{aggregated_key}"] = mean
            summary[f"std_{aggregated_key}"] = std
            summary[f"{aggregated_key}_ci95_low"] = low
            summary[f"{aggregated_key}_ci95_high"] = high
        summary["n_saliency_seeds"] = sum(
            "saliency_return_delta" in result for result in results
        )
        summaries.append(summary)
    return inventory, summaries, failed


def _standardized_paired_effect(differences: np.ndarray) -> float | None:
    """Return Cohen's dz (paired mean difference / paired-difference SD)."""

    if differences.size < 2:
        return None
    standard_deviation = float(np.std(differences, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation == 0.0:
        return None
    return float(np.mean(differences) / standard_deviation)


def _paired_statistics(
    differences: Sequence[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    array = np.asarray(differences, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None, None, None, None, None, None

    mean_difference, difference_low, difference_high = bootstrap_mean_ci(
        array,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    effect = _standardized_paired_effect(array)
    if effect is None:
        return mean_difference, difference_low, difference_high, None, None, None

    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(0, array.size, size=(bootstrap_samples, array.size))
    effects = [
        candidate
        for sample in array[indices]
        if (candidate := _standardized_paired_effect(sample)) is not None
        and math.isfinite(candidate)
    ]
    if not effects:
        return mean_difference, difference_low, difference_high, effect, None, None
    effect_low, effect_high = np.quantile(np.asarray(effects), [0.025, 0.975])
    return (
        mean_difference,
        difference_low,
        difference_high,
        effect,
        float(effect_low),
        float(effect_high),
    )


def paired_model_comparisons(
    per_run: Sequence[Mapping[str, Any]],
    *,
    reference_model: str = "comm_identity",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    """Compute matched-seed model-minus-reference effects.

    Pairing is restricted to the same task, algorithm, ablation, and ablation
    value. Expected pairs come from the manifest inventory, while statistics
    use one completed attempt per seed.
    """

    grouped: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in per_run:
        required = ("task", "algorithm", "model", "seed")
        if all(row.get(key) is not None for key in required):
            grouped[_comparison_key(row)][str(row["model"])].append(row)

    comparisons: list[dict[str, Any]] = []
    metrics = ("mean_final_return", "normalized_return_auc")
    for comparison_key, by_model in sorted(grouped.items(), key=lambda item: str(item[0])):
        if reference_model not in by_model:
            continue
        reference_rows = by_model[reference_model]
        reference_expected = {
            int(row["seed"]) for row in reference_rows if row.get("expected_in_manifest")
        }
        if not reference_expected:
            reference_expected = {int(row["seed"]) for row in reference_rows}
        reference_has_manifest = any(row.get("expected_in_manifest") for row in reference_rows)
        reference_completed = _completed_seed_rows(reference_rows)

        for model, model_rows in sorted(by_model.items()):
            if model == reference_model:
                continue
            model_expected = {
                int(row["seed"]) for row in model_rows if row.get("expected_in_manifest")
            }
            if not model_expected:
                model_expected = {int(row["seed"]) for row in model_rows}
            model_has_manifest = any(row.get("expected_in_manifest") for row in model_rows)
            expected_pair_seeds = model_expected & reference_expected
            model_completed = _completed_seed_rows(model_rows)

            for metric in metrics:
                candidate_pair_seeds = model_completed.keys() & reference_completed.keys()
                if reference_has_manifest or model_has_manifest:
                    candidate_pair_seeds &= expected_pair_seeds
                paired_seeds = sorted(
                    seed
                    for seed in candidate_pair_seeds
                    if model_completed[seed].get(metric) is not None
                    and reference_completed[seed].get(metric) is not None
                    and math.isfinite(float(model_completed[seed][metric]))
                    and math.isfinite(float(reference_completed[seed][metric]))
                )
                model_values = [float(model_completed[seed][metric]) for seed in paired_seeds]
                reference_values = [
                    float(reference_completed[seed][metric]) for seed in paired_seeds
                ]
                differences = [
                    model_value - reference_value
                    for model_value, reference_value in zip(
                        model_values, reference_values, strict=True
                    )
                ]
                (
                    mean_difference,
                    difference_low,
                    difference_high,
                    effect,
                    effect_low,
                    effect_high,
                ) = _paired_statistics(
                    differences,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
                comparisons.append(
                    {
                        "task": comparison_key[0],
                        "algorithm": comparison_key[1],
                        "ablation": comparison_key[2],
                        "ablation_value": comparison_key[3],
                        "model": model,
                        "reference_model": reference_model,
                        "direction": "model_minus_reference",
                        "metric": metric,
                        "n_expected_pairs": len(expected_pair_seeds),
                        "expected_pair_seeds": ";".join(
                            str(seed) for seed in sorted(expected_pair_seeds)
                        ),
                        "n_pairs": len(paired_seeds),
                        "paired_seeds": ";".join(str(seed) for seed in paired_seeds),
                        "missing_pair_seeds": ";".join(
                            str(seed) for seed in sorted(expected_pair_seeds - set(paired_seeds))
                        ),
                        "model_only_completed_seeds": ";".join(
                            str(seed)
                            for seed in sorted(model_completed.keys() - reference_completed.keys())
                        ),
                        "reference_only_completed_seeds": ";".join(
                            str(seed)
                            for seed in sorted(reference_completed.keys() - model_completed.keys())
                        ),
                        "paired_model_mean": (
                            float(np.mean(model_values)) if model_values else None
                        ),
                        "paired_reference_mean": (
                            float(np.mean(reference_values)) if reference_values else None
                        ),
                        "mean_paired_difference": mean_difference,
                        "difference_ci95_low": difference_low,
                        "difference_ci95_high": difference_high,
                        "paired_effect_size_cohens_dz": effect,
                        "effect_size_ci95_low": effect_low,
                        "effect_size_ci95_high": effect_high,
                    }
                )
    return comparisons


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    keys = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        if not keys:
            file.write("")
            return
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(materialized)


def _series_label(metadata: Mapping[str, Any]) -> str:
    model = str(metadata["model"])
    ablation = metadata.get("ablation")
    if ablation not in {None, "", "main"}:
        return f"{model} | {ablation}={metadata.get('ablation_value')}"
    return model


def _ablation_value_sort_key(value: Any) -> tuple[int, float | str]:
    """Sort numeric sweep values numerically and named controls lexically."""

    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _load_curves(suite_dir: Path) -> dict[str, list[list[tuple[int, float]]]]:
    by_seed: dict[str, dict[int, tuple[int, list[tuple[int, float]]]]] = defaultdict(dict)
    for run_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _read_json(metadata_path)
        if _status_from_run_dir(run_dir, metadata) == "completed":
            curve = _evaluation_curve(_metric_rows(run_dir / "metrics.csv"))
            if curve:
                label = _series_label(metadata)
                seed = int(metadata["seed"])
                attempt = _as_int(metadata.get("attempt"))
                previous = by_seed[label].get(seed)
                if previous is None or attempt >= previous[0]:
                    by_seed[label][seed] = (attempt, curve)
    return {
        label: [seed_curves[seed][1] for seed in sorted(seed_curves)]
        for label, seed_curves in by_seed.items()
    }


def _make_plots(
    suite_dir: Path,
    results_dir: Path,
    per_run: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> tuple[Path, ...]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Plotting requires the analysis extra: uv sync --extra analysis"
        ) from error

    outputs: list[Path] = []
    curves = _load_curves(suite_dir)
    if curves:
        figure, axis = plt.subplots(figsize=(8, 5))
        for model, model_curves in sorted(curves.items()):
            shared_frames = sorted(set.intersection(*(set(dict(curve)) for curve in model_curves)))
            if not shared_frames:
                continue
            matrix = np.asarray(
                [[dict(curve)[frame] for frame in shared_frames] for curve in model_curves]
            )
            mean = matrix.mean(axis=0)
            axis.plot(shared_frames, mean, label=model)
            if matrix.shape[0] > 1:
                generator = np.random.default_rng(0)
                indices = generator.integers(0, matrix.shape[0], size=(2_000, matrix.shape[0]))
                bootstrapped = matrix[indices].mean(axis=1)
                low, high = np.quantile(bootstrapped, [0.025, 0.975], axis=0)
                axis.fill_between(
                    shared_frames,
                    low,
                    high,
                    alpha=0.2,
                )
        axis.set(xlabel="Environment frames", ylabel="Evaluation return")
        axis.legend()
        figure.tight_layout()
        path = results_dir / "learning_curves.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)

    final_summaries = [
        row
        for row in summaries
        if row.get("mean_final_return") is not None
        and math.isfinite(float(row["mean_final_return"]))
    ]
    if final_summaries:
        labels = [_series_label(row) for row in final_summaries]
        means = [float(row["mean_final_return"]) for row in final_summaries]
        low = [
            max(0.0, mean - float(row["ci95_low"]))
            for mean, row in zip(means, final_summaries, strict=True)
        ]
        high = [
            max(0.0, float(row["ci95_high"]) - mean)
            for mean, row in zip(means, final_summaries, strict=True)
        ]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(labels, means, yerr=np.asarray([low, high]), capsize=4)
        axis.set(ylabel="Mean final evaluation return")
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        path = results_dir / "final_performance.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)

        auc_summaries = [
            row
            for row in summaries
            if row.get("mean_normalized_auc") is not None
            and math.isfinite(float(row["mean_normalized_auc"]))
        ]
        if auc_summaries:
            auc_labels = [_series_label(row) for row in auc_summaries]
            auc_means = [float(row["mean_normalized_auc"]) for row in auc_summaries]
            auc_low = [
                max(0.0, mean - float(row["normalized_auc_ci95_low"]))
                for mean, row in zip(auc_means, auc_summaries, strict=True)
            ]
            auc_high = [
                max(0.0, float(row["normalized_auc_ci95_high"]) - mean)
                for mean, row in zip(auc_means, auc_summaries, strict=True)
            ]
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.bar(
                auc_labels,
                auc_means,
                yerr=np.asarray([auc_low, auc_high]),
                capsize=4,
            )
            axis.set(ylabel="Normalized return AUC")
            axis.tick_params(axis="x", rotation=35)
            figure.tight_layout()
            path = results_dir / "sample_efficiency.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            outputs.append(path)

        time_summaries = [row for row in summaries if row.get("mean_training_time") is not None]
        if time_summaries:
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.bar(
                [_series_label(row) for row in time_summaries],
                [float(row["mean_training_time"]) for row in time_summaries],
            )
            axis.set(ylabel="Training wall-clock seconds")
            axis.tick_params(axis="x", rotation=35)
            figure.tight_layout()
            path = results_dir / "wall_clock.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            outputs.append(path)

    cost_key = next(
        (
            key
            for key in (
                "comm_realized_bits_per_step",
                "comm_nominal_bits_per_step",
                "comm_realized_sender_bits_per_step",
                "comm_message_bits_per_sender",
                "comm_message_bits",
            )
            if any(key in row for row in per_run)
        ),
        None,
    )
    cost_summaries = [
        row
        for row in summaries
        if cost_key is not None
        and row.get(f"mean_{cost_key}") is not None
        and row.get("mean_final_return") is not None
    ]
    if cost_key is not None and cost_summaries:
        figure, axis = plt.subplots(figsize=(7, 5))
        for row in cost_summaries:
            cost_mean = float(row[f"mean_{cost_key}"])
            return_mean = float(row["mean_final_return"])
            x_error = np.asarray(
                [
                    [max(0.0, cost_mean - float(row[f"{cost_key}_ci95_low"]))],
                    [max(0.0, float(row[f"{cost_key}_ci95_high"]) - cost_mean)],
                ]
            )
            y_error = np.asarray(
                [
                    [max(0.0, return_mean - float(row["ci95_low"]))],
                    [max(0.0, float(row["ci95_high"]) - return_mean)],
                ]
            )
            axis.errorbar(
                cost_mean,
                return_mean,
                xerr=x_error,
                yerr=y_error,
                marker="o",
                capsize=4,
                linestyle="none",
                label=_series_label(row),
            )
        axis.set(xlabel=cost_key, ylabel="Mean final evaluation return")
        axis.legend()
        figure.tight_layout()
        path = results_dir / "communication_efficiency.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)

    saliency_summaries = [
        row
        for row in summaries
        if row.get("mean_saliency_return_delta") is not None
        and math.isfinite(float(row["mean_saliency_return_delta"]))
    ]
    if saliency_summaries:
        figure, axis = plt.subplots(figsize=(8, 5))
        labels = [_series_label(row) for row in saliency_summaries]
        means = [float(row["mean_saliency_return_delta"]) for row in saliency_summaries]
        low = [
            max(0.0, mean - float(row["saliency_return_delta_ci95_low"]))
            for mean, row in zip(means, saliency_summaries, strict=True)
        ]
        high = [
            max(0.0, float(row["saliency_return_delta_ci95_high"]) - mean)
            for mean, row in zip(means, saliency_summaries, strict=True)
        ]
        axis.bar(labels, means, yerr=np.asarray([low, high]), capsize=4)
        axis.axhline(0.0, linewidth=1, color="black")
        axis.set(
            ylabel="Return lost when communication is severed",
            title="Communication saliency (higher = channel matters more)",
        )
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        path = results_dir / "communication_saliency.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)

    ablations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        if row.get("ablation") not in {None, "", "main"}:
            ablations[str(row["ablation"])].append(row)
    for ablation, rows in sorted(ablations.items()):
        figure, axis = plt.subplots(figsize=(7, 5))
        plotted = False
        by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_model[str(row["model"])].append(row)
        for model, model_rows in sorted(by_model.items()):
            model_rows = sorted(
                (row for row in model_rows if row.get("mean_final_return") is not None),
                key=lambda row: _ablation_value_sort_key(row["ablation_value"]),
            )
            if not model_rows:
                continue
            plotted = True
            means = [float(row["mean_final_return"]) for row in model_rows]
            axis.errorbar(
                [str(row["ablation_value"]) for row in model_rows],
                means,
                yerr=np.asarray(
                    [
                        [
                            max(0.0, mean - float(row["ci95_low"]))
                            for mean, row in zip(means, model_rows, strict=True)
                        ],
                        [
                            max(0.0, float(row["ci95_high"]) - mean)
                            for mean, row in zip(means, model_rows, strict=True)
                        ],
                    ]
                ),
                marker="o",
                capsize=4,
                label=model,
            )
        if not plotted:
            plt.close(figure)
            continue
        axis.set(xlabel=ablation, ylabel="Mean final evaluation return")
        axis.legend()
        figure.tight_layout()
        filename = re.sub(r"[^a-z0-9_-]+", "_", ablation.lower())
        path = results_dir / f"return_vs_{filename}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)
    return tuple(outputs)


def analyze_suite(
    suite_dir: Path,
    *,
    results_dir: Path | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    plots: bool = True,
) -> AnalysisOutputs:
    suite_dir = suite_dir.resolve()
    destination = results_dir.resolve() if results_dir is not None else suite_dir / "results"
    destination.mkdir(parents=True, exist_ok=True)
    per_run, summaries, failed = aggregate_suite(
        suite_dir,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    comparisons = paired_model_comparisons(
        per_run,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    prefix = suite_dir.name
    per_run_path = destination / f"{prefix}_per_run.csv"
    summary_path = destination / f"{prefix}_summary.csv"
    failed_path = destination / f"{prefix}_failed_runs.csv"
    comparisons_path = destination / f"{prefix}_paired_comparisons.csv"
    _write_rows(per_run_path, per_run)
    _write_rows(summary_path, summaries)
    _write_rows(failed_path, failed)
    _write_rows(comparisons_path, comparisons)
    plot_files = _make_plots(suite_dir, destination, per_run, summaries) if plots else ()
    return AnalysisOutputs(
        per_run_csv=per_run_path,
        summary_csv=summary_path,
        failed_runs_csv=failed_path,
        plot_files=plot_files,
        paired_comparisons_csv=comparisons_path,
    )
