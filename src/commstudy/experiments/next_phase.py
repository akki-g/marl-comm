"""Prospective, method-blind allocation rules for the next diagnostic phase.

These rules select a common finite budget; they never certify convergence or
select experiments based on a positive communication effect.
"""

from __future__ import annotations

import hashlib
import json
import math


MAPDN_CALIBRATION_SEEDS = (40, 41)
MAPDN_COMPARISON_SEEDS = (50, 51, 52, 53, 54)
MAPDN_CHECKPOINTS = (0, 8192, 16384, 24576, 32768)
MAPDN_METHODS = ("identity", "local_capacity", "broadcast")
MAPDN_HORIZON = 239


def content_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def select_nonoverlapping_rows(starts, count, *, horizon, history, excluded=()):
    """Choose central quantiles of legal runs, excluding full data footprints.

    `excluded` contains half-open absolute-row intervals. Unlike random seeds,
    interval exclusion actually prevents reuse of previously inspected windows.
    """
    if type(count) is not int or count < 1 or horizon < 1 or history < 1:
        raise ValueError("Positive count, horizon and history required")
    starts = sorted(starts)
    if not starts or any(type(row) is not int for row in starts):
        raise ValueError("Manifest starts must be nonempty integer rows")
    if len(set(starts)) != len(starts):
        raise ValueError("Duplicate manifest starts")
    if any(a >= b for a, b in excluded):
        raise ValueError("Excluded footprints must be nonempty half-open intervals")
    candidates = [
        row
        for row in starts
        if all(row + horizon + 1 <= low or row - history + 1 >= high for low, high in excluded)
    ]
    if len(candidates) < count:
        raise ValueError("Insufficient unused episode windows")
    chosen = [
        candidates[((2 * index + 1) * len(candidates)) // (2 * count)] for index in range(count)
    ]
    if any(
        right - history + 1 < left + horizon + 1
        for left, right in zip(chosen[:-1], chosen[1:], strict=True)
    ):
        raise ValueError("Selected complete episode footprints overlap")
    return chosen


def choose_mapdn_budget(calibration):
    """Consume exactly two Identity-only trajectories with fixed checkpoints.

    A stable allocation needs two successive small validation changes for both
    seeds. The cap is selected if no candidate meets the rule; this is explicitly
    an unresolved adequacy result, not grounds to tune until communication wins.
    """
    if len(calibration) != len(MAPDN_CALIBRATION_SEEDS):
        raise ValueError("Both declared baseline training seeds are required")
    by_seed = {}
    keys = ("return", "voltage_magnitude", "violation_frequency", "q_effort")
    for run in calibration:
        if run.get("method") != "identity" or run.get("seed") in by_seed:
            raise ValueError("Budget calibration accepts unique Identity baselines only")
        if run.get("integrity_passed") is not True:
            raise ValueError("Failed baseline integrity blocks comparison allocation")
        points = run["validation"]
        if tuple(point["frames"] for point in points) != MAPDN_CHECKPOINTS:
            raise ValueError("Missing, reordered or unexpected validation checkpoints")
        for point in points:
            if point.get("complete") is not True or point.get("solver_failures") != 0:
                raise ValueError("Budget validation requires complete solver-feasible banks")
            if any(
                not isinstance(point.get(key), (int, float)) or not math.isfinite(point[key])
                for key in keys
            ):
                raise ValueError("Nonfinite or missing validation metric")
        by_seed[run["seed"]] = points
    if set(by_seed) != set(MAPDN_CALIBRATION_SEEDS):
        raise ValueError("Unexpected calibration training seeds")
    records = []
    for index in range(2, len(MAPDN_CHECKPOINTS)):
        checks = []
        for seed, points in sorted(by_seed.items()):
            tolerance = {
                "return": max(1.0, 0.05 * abs(points[0]["return"])),
                "voltage_magnitude": 0.00005,
                "violation_frequency": 0.005,
                "q_effort": 0.02,
            }
            for end in (index - 1, index):
                changes = {key: abs(points[end][key] - points[end - 1][key]) for key in keys}
                checks.append(
                    {
                        "seed": seed,
                        "from_frames": points[end - 1]["frames"],
                        "to_frames": points[end]["frames"],
                        "changes": changes,
                        "tolerances": tolerance,
                        "stable": all(changes[key] <= tolerance[key] for key in keys),
                    }
                )
        records.append(
            {
                "frames": MAPDN_CHECKPOINTS[index],
                "checks": checks,
                "stable": all(check["stable"] for check in checks),
            }
        )
    stable = [row for row in records if row["stable"]]
    return {
        "selected_frames": stable[0]["frames"] if stable else MAPDN_CHECKPOINTS[-1],
        "validation_stability_established": bool(stable),
        "selection_reason": "first_declared_stable_budget" if stable else "declared_compute_cap",
        "convergence_established": False,
        "baseline_training_seeds": sorted(by_seed),
        "method_outcomes_used": ["identity"],
        "candidate_checks": records,
        "interpretation": "A finite common allocation; stability is not policy optimality.",
    }


def paired_seed_interval(values, *, bootstrap_seed=9132026, samples=10000):
    """Descriptive paired-training-seed bootstrap; episodes are not replicates."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("Exactly five finite paired-training-seed differences required")
    draws = np.random.default_rng(bootstrap_seed).choice(values, (samples, len(values)))
    low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "interval_95": [float(low), float(high)],
        "seed_values": values.tolist(),
        "n_training_seeds": len(values),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": samples,
        "inference": "descriptive; no multiplicity adjustment or population guarantee",
    }
