"""Read preserved MAPDN evidence and write descriptive analysis outside its directory.

Standard library only: this program cannot construct environments, load a
policy, optimize a model, or change the recorded scientific verdict.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys


PHYSICAL = {
    "percentage_of_v_out_of_control": "fraction",
    "percentage_of_lower_than_lower_v": "fraction",
    "percentage_of_higher_than_upper_v": "fraction",
    "totally_controllable_ratio": "fraction",
    "average_voltage_deviation": "per_unit",
    "average_voltage": "per_unit",
    "max_voltage_drop_deviation": "per_unit",
    "max_voltage_rise_deviation": "per_unit",
    "total_line_loss": "MW",
    "q_loss": "MVAr",
    "voltage_violation_magnitude": "per_unit",
    "voltage_bus_count": "buses",
    "voltage_violating_bus_count": "buses",
}
ATTEMPTED = {
    "destroy": "indicator",
    "action_solver_failure": "indicator",
    "advance_solver_failure": "indicator",
    "pv_curtailed_mw": "MW",
    "feasible_step": "indicator",
    "reward": "reward_units",
    "voltage_metrics_valid": "indicator",
    "valid": "indicator",
}
REQUIRED_HEALTH = {
    "collection": (
        "action_finite_fraction",
        "loc_finite_fraction",
        "scale_finite_fraction",
        "replay_accepted_log_prob_max_tolerance_ratio",
        "replay_log_prob_max_abs_error",
        "replay_transition_count",
        "replay_collection_shape_fallback_count",
    ),
    "training": (
        "advantage_finite_fraction",
        "state_value_finite_fraction",
        "value_target_finite_fraction",
        "loss_objective",
        "loss_critic",
        "loss_entropy",
        "grad_norm_loss_objective",
        "grad_norm_loss_critic",
        "kl_approx",
        "ESS",
        "entropy",
        "clip_fraction",
        "optimization_transition_count",
    ),
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha(value):
    value = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.records = {}
        self.recorded_root = self.root

    def path(self, value):
        path = Path(value)
        if path.is_absolute() and not path.is_relative_to(self.root):
            try:
                path = self.root / path.relative_to(self.recorded_root)
            except ValueError as error:
                raise ValueError(
                    f"Evidence reference escapes the input directory: {path}"
                ) from error
        elif not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Evidence reference resolves outside the input directory: {path}")
        return path

    def _register(self, path, digest, role):
        name = path.relative_to(self.root).as_posix()
        previous = self.records.get(name)
        if previous is not None and previous["sha256"] != digest:
            raise ValueError(f"Input changed between reads: {name}")
        self.records[name] = {"sha256": digest, "size_bytes": path.stat().st_size, "role": role}

    def read(self, value, role):
        path = self.path(value)
        data = path.read_bytes()
        self._register(path, hashlib.sha256(data).hexdigest(), role)
        return data

    def json(self, value, role):
        return json.loads(self.read(value, role))

    def optional_json(self, value, role):
        return self.json(value, role) if self.path(value).is_file() else None

    def hash(self, value, role):
        path = self.path(value)
        digest = file_sha(path)
        self._register(path, digest, role)
        return digest

    def verify_unchanged(self):
        for name, record in self.records.items():
            if file_sha(self.path(name)) != record["sha256"]:
                raise ValueError(f"Evidence input changed during analysis: {name}")


def numeric(value):
    if not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def describe(values, *, eligible, missing=0, nonfinite=0):
    finite = [float(value) for value in values]
    mean = math.fsum(finite) / len(finite) if finite else None
    return {
        "eligible_denominator": eligible,
        "finite_value_denominator": len(finite),
        "missing_values": missing,
        "nonfinite_values": nonfinite,
        "mean": mean if not missing and not nonfinite and len(finite) == eligible else None,
        "mean_over_finite_values": mean,
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "sum_over_finite_values": math.fsum(finite) if finite else None,
    }


def action_summary(transitions, n_inverters):
    values = []
    missing = invalid = nonfinite = 0
    for step in transitions:
        actions = step.get("actions")
        if actions is None:
            missing += n_inverters
            continue
        if (
            not isinstance(actions, list)
            or len(actions) != n_inverters
            or any(not isinstance(value, list) or len(value) != 1 for value in actions)
        ):
            invalid += 1
            continue
        for value in actions:
            number = numeric(value[0])
            if number is None:
                nonfinite += 1
            else:
                values.append(abs(number))
    saturated = sum(value / 0.8 > 0.999 for value in values)
    expected = len(transitions) * n_inverters
    return {
        "physical_bounds": [-0.8, 0.8],
        "saturation_rule": "abs(action)/0.8 > 0.999",
        "expected_action_scalars": expected,
        "finite_action_scalar_denominator": len(values),
        "denominator_formula": f"{len(transitions)} transitions * {n_inverters} inverters",
        "missing_action_scalars": missing,
        "invalid_shape_transitions": invalid,
        "nonfinite_or_nonnumeric_action_scalars": nonfinite,
        "saturated_action_scalars": saturated,
        "saturated_fraction_over_finite_scalars": saturated / len(values) if values else None,
        "complete_scalar_coverage": len(values) == expected,
        "mean_absolute_action": math.fsum(values) / len(values) if values else None,
        "maximum_absolute_action": max(values) if values else None,
    }


def transition_summary(transitions, *, n_inverters, include_actions=True):
    physical = [
        step
        for step in transitions
        if step.get("diagnostics", {}).get("valid") == 1
        and step.get("diagnostics", {}).get("voltage_metrics_valid") == 1
    ]
    metrics = {}
    for name, unit in (PHYSICAL | ATTEMPTED).items():
        selected = physical if name in PHYSICAL else transitions
        values, missing, nonfinite = [], 0, 0
        for step in selected:
            diagnostics = step.get("diagnostics", {})
            if name not in diagnostics or diagnostics[name] is None:
                missing += 1
            elif numeric(diagnostics[name]) is None:
                nonfinite += 1
            else:
                values.append(float(diagnostics[name]))
        metrics[name] = {
            "unit": unit,
            "scope": "physical_valid" if name in PHYSICAL else "all_attempted",
            **describe(values, eligible=len(selected), missing=missing, nonfinite=nonfinite),
        }
    violating = buses = complete_counts = invalid_counts = 0
    for step in physical:
        diagnostics = step.get("diagnostics", {})
        n, k = (
            numeric(diagnostics.get("voltage_bus_count")),
            numeric(diagnostics.get("voltage_violating_bus_count")),
        )
        if n is None or k is None or n <= 0 or n != int(n) or k != int(k) or not 0 <= k <= n:
            invalid_counts += 1
        else:
            complete_counts += 1
            violating += int(k)
            buses += int(n)
    missing_rows = noninteger_rows = alignment_errors = 0
    for step in transitions:
        diagnostics = step.get("diagnostics", {})
        reward_row = numeric(diagnostics.get("reward_row"))
        next_row = numeric(diagnostics.get("observation_row"))
        if reward_row is None or next_row is None:
            missing_rows += 1
        elif reward_row != int(reward_row) or next_row != int(next_row):
            noninteger_rows += 1
        elif diagnostics.get("destroy") == 0 and next_row != reward_row + 1:
            alignment_errors += 1
    return {
        "attempted_transition_denominator": len(transitions),
        "valid_transport_transitions": sum(
            step.get("diagnostics", {}).get("valid") == 1 for step in transitions
        ),
        "physical_valid_transition_denominator": len(physical),
        "physical_excluded_transitions": len(transitions) - len(physical),
        "row_diagnostics": {
            "attempted_transition_denominator": len(transitions),
            "missing_or_nonfinite_row_transitions": missing_rows,
            "noninteger_row_transitions": noninteger_rows,
            "healthy_step_alignment_errors": alignment_errors,
        },
        "metrics": metrics,
        "voltage_frequency": {
            "violated_bus_count": violating,
            "bus_transition_denominator": buses,
            "complete_count_transitions": complete_counts,
            "invalid_count_transitions": invalid_counts,
            "frequency": violating / buses if buses and not invalid_counts else None,
        },
        "actions": action_summary(transitions, n_inverters)
        if include_actions
        else {"available": False, "reason": "Training transition evidence does not retain actions"},
    }


def bank_summary(bank, *, n_inverters):
    episodes, all_steps = [], []
    for item in bank.get("episodes", []):
        steps = item.get("transitions", [])
        all_steps.extend(steps)
        rewards = [numeric(step.get("reward")) for step in steps]
        return_value = (
            math.fsum(rewards) if rewards and all(v is not None for v in rewards) else None
        )
        row_errors = 0
        for index, step in enumerate(steps):
            diagnostics = step.get("diagnostics", {})
            row = item.get("start_row")
            if not isinstance(row, int) or diagnostics.get("reward_row") != row + index:
                row_errors += 1
        episodes.append(
            {
                "episode_id": item.get("episode_id"),
                "start_row": item.get("start_row"),
                "seed": item.get("seed"),
                "complete": item.get("complete") is True,
                "terminated": bool(steps and steps[-1].get("terminated")),
                "truncated": bool(steps and steps[-1].get("truncated")),
                "transition_count": len(steps),
                "saved_transition_count": item.get("transition_count"),
                "return": return_value,
                "saved_return": numeric(item.get("return")),
                "nonfinite_or_missing_step_rewards": sum(value is None for value in rewards),
                "reward_row_alignment_errors": row_errors,
                "domain": transition_summary(steps, n_inverters=n_inverters),
            }
        )
    returns = [episode["return"] for episode in episodes if episode["return"] is not None]
    return {
        "split": bank.get("split"),
        "policy": bank.get("policy"),
        "declared_bank": bank.get("bank"),
        "episodes": episodes,
        "episode_count": len(episodes),
        "complete_episodes": sum(e["complete"] for e in episodes),
        "episode_returns": describe(
            returns, eligible=len(episodes), missing=len(episodes) - len(returns)
        ),
        "pooled_domain": transition_summary(all_steps, n_inverters=n_inverters),
    }


def _difference(before, after):
    return None if before is None or after is None else after - before


def pair_banks(initial, final):
    def index(bank):
        result, duplicates = {}, []
        for episode in bank["episodes"]:
            key = (episode["episode_id"], episode["start_row"], episode["seed"])
            if key in result:
                duplicates.append(list(key))
            result[key] = episode
        return result, duplicates

    first, first_duplicates = index(initial)
    last, last_duplicates = index(final)
    pairs = []
    for key in sorted(set(first) | set(last), key=repr):
        before, after = first.get(key), last.get(key)
        if before is None or after is None:
            pairs.append(
                {
                    "identity": list(key),
                    "paired": False,
                    "missing": "initial" if before is None else "final",
                }
            )
            continue
        pairs.append(
            {
                "identity": list(key),
                "paired": True,
                "initial_complete": before["complete"],
                "final_complete": after["complete"],
                "initial_transitions": before["transition_count"],
                "final_transitions": after["transition_count"],
                "initial_physical_denominator": before["domain"][
                    "physical_valid_transition_denominator"
                ],
                "final_physical_denominator": after["domain"][
                    "physical_valid_transition_denominator"
                ],
                "return_change_final_minus_initial": _difference(before["return"], after["return"]),
                "metric_mean_changes_final_minus_initial": {
                    name: _difference(
                        before["domain"]["metrics"][name]["mean"],
                        after["domain"]["metrics"][name]["mean"],
                    )
                    for name in PHYSICAL | ATTEMPTED
                },
                "voltage_frequency_change_final_minus_initial": _difference(
                    before["domain"]["voltage_frequency"]["frequency"],
                    after["domain"]["voltage_frequency"]["frequency"],
                ),
                "physical_action_saturation_change_final_minus_initial": _difference(
                    before["domain"]["actions"]["saturated_fraction_over_finite_scalars"],
                    after["domain"]["actions"]["saturated_fraction_over_finite_scalars"],
                ),
            }
        )
    paired = [pair for pair in pairs if pair["paired"]]
    return {
        "identity_fields": ["episode_id", "start_row", "seed"],
        "pairing_valid": not first_duplicates and not last_duplicates and len(paired) == len(pairs),
        "initial_duplicate_identities": first_duplicates,
        "final_duplicate_identities": last_duplicates,
        "paired_episodes": len(paired),
        "unpaired_episodes": len(pairs) - len(paired),
        "episodes": pairs,
        "mean_paired_metric_changes": {
            name: describe(
                [
                    pair["metric_mean_changes_final_minus_initial"][name]
                    for pair in paired
                    if pair["metric_mean_changes_final_minus_initial"][name] is not None
                ],
                eligible=len(paired),
                missing=sum(
                    pair["metric_mean_changes_final_minus_initial"][name] is None for pair in paired
                ),
            )
            for name in PHYSICAL | ATTEMPTED
        },
    }


def training_csv_summary(data, expected_frames):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    series = defaultdict(list)
    malformed, nonfinite, missing_values = [], [], []
    for index, row in enumerate(rows):
        identity = (row.get("phase"), row.get("group"), row.get("metric"))
        try:
            frame = int(row["frames"])
        except (KeyError, TypeError, ValueError):
            malformed.append(index + 2)
            continue
        raw = row.get("value")
        try:
            value = float(raw) if raw not in {None, ""} else None
        except (TypeError, ValueError):
            value = None
        if value is None:
            missing_values.append(index + 2)
        elif not math.isfinite(value):
            nonfinite.append(index + 2)
            value = None
        series[identity].append((frame, value))
    health = {}
    for phase, names in REQUIRED_HEALTH.items():
        for name in names:
            observations = series[(phase, "agents", name)]
            frames = [frame for frame, _ in observations]
            values = [value for _, value in observations if value is not None]
            health[f"{phase}/agents/{name}"] = {
                **describe(
                    values, eligible=len(observations), missing=len(observations) - len(values)
                ),
                "expected_frame_count": len(expected_frames),
                "observed_frames": frames,
                "missing_frames": sorted(set(expected_frames) - set(frames)),
                "unexpected_frames": sorted(set(frames) - set(expected_frames)),
                "duplicate_frame_count": len(frames) - len(set(frames)),
                "below_one_finite_fraction_rows": sum(value < 1 for value in values)
                if name.endswith("finite_fraction")
                else None,
                "over_one_replay_tolerance_rows": sum(value > 1 for value in values)
                if name == "replay_accepted_log_prob_max_tolerance_ratio"
                else None,
            }
    return {
        "rows": len(rows),
        "malformed_frame_csv_lines": malformed,
        "nonfinite_value_csv_lines": nonfinite,
        "missing_or_invalid_value_csv_lines": missing_values,
        "required_health_series": health,
        "generic_action_saturated_fraction_used_for_physical_saturation": False,
    }


def summarize(root):
    inputs = Inputs(root)
    recorded = inputs.json("report.json", "original protocol verdict and retention receipts")
    packet = inputs.json("preparation.json", "frozen preparation/source/runtime hashes")
    inputs.recorded_root = Path(packet["output"])
    manifest = inputs.json("split_manifest.json", "data provenance and temporal split manifest")
    normalizer = inputs.json("normalizer.json", "fixed normalization provenance")
    inputs.json("banks.json", "declared fixed episode banks")
    inputs.read("frozen_plan.md", "frozen scientific plan")
    issues = []
    if packet.get("sha256") != _content_sha(packet):
        issues.append("preparation content digest differs")
    if recorded.get("preparation_sha256") != packet.get("sha256"):
        issues.append("original verdict references another preparation digest")
    launch = inputs.optional_json("launch.json", "launch provenance")
    if launch is not None and launch.get("preparation_sha256") != packet.get("sha256"):
        issues.append("launch preparation digest differs")
    inputs.optional_json("training_complete.json", "test-isolation completion marker")
    n_inverters = packet["feeder_dimensions"]["inverters"]
    rows = {item["seed"]: item for item in recorded.get("training", [])}
    seeds = {}
    for seed in packet["allocation"]:
        spec = inputs.json(f"spec_seed{seed}.json", f"seed {seed} frozen resolved configuration")
        evidence = Path("evidence") / f"seed{seed}"
        seed_result = {"banks": {}, "missing_files": [], "training": {}}
        for split in ("validation", "test"):
            banks = {}
            for label in ("initial", "final"):
                name = evidence / f"{label}_{split}.json"
                raw = inputs.optional_json(
                    name, f"seed {seed} {label} {split} complete transition bank"
                )
                if raw is None:
                    seed_result["missing_files"].append(name.as_posix())
                else:
                    banks[label] = bank_summary(raw, n_inverters=n_inverters)
            seed_result["banks"][split] = banks
            if set(banks) == {"initial", "final"}:
                seed_result["banks"][split]["paired_effects"] = pair_banks(
                    banks["initial"], banks["final"]
                )
        summary = inputs.optional_json(
            evidence / "training_summary.json", "recorded training checks"
        )
        seed_result["training"]["recorded_summary"] = summary
        transition_path = evidence / "training_transitions.jsonl"
        if inputs.path(transition_path).is_file():
            records = [
                json.loads(line)
                for line in inputs.read(
                    transition_path, "training per-transition diagnostics"
                ).splitlines()
                if line.strip()
            ]
            frame_ids = [record.get("frame") for record in records]
            seed_result["training"]["transitions"] = transition_summary(
                records, n_inverters=n_inverters, include_actions=False
            )
            seed_result["training"]["sequential_frame_ids"] = frame_ids == list(
                range(1, len(records) + 1)
            )
            seed_result["training"]["expected_frames"] = spec["experiment"]["max_n_frames"]
        else:
            seed_result["missing_files"].append(transition_path.as_posix())
        row = rows.get(seed) or summary
        run_dir = (
            inputs.path(row["run_dir"])
            if row
            else inputs.path(Path("runs") / packet["protocol_id"] / f"identity_seed{seed}")
        )
        for name in ("metadata.json", "status.json"):
            seed_result["training"][name.removesuffix(".json")] = inputs.optional_json(
                run_dir / name, f"managed training {name}"
            )
        if (run_dir / "metrics.csv").is_file():
            batch = spec["experiment"]["on_policy_collected_frames_per_batch"]
            frame_limit = spec["experiment"]["max_n_frames"]
            seed_result["training"]["tidy_metrics"] = training_csv_summary(
                inputs.read(run_dir / "metrics.csv", "tidy training/collection health metrics"),
                list(range(batch, frame_limit + 1, batch)),
            )
        else:
            seed_result["missing_files"].append(
                str((run_dir / "metrics.csv").relative_to(inputs.root))
            )
        seeds[str(seed)] = seed_result
    archive_checks = []
    receipts = [recorded.get("evidence_archive")]
    for row in recorded.get("training", []):
        receipts.extend(
            row.get(name)
            for name in ("managed_run_archive", "pretest_initial_and_training_archive")
        )
    for receipt in filter(None, receipts):
        path = inputs.path(receipt["path"])
        actual = inputs.hash(path, "retained archive bytes") if path.is_file() else None
        archive_checks.append(
            {
                "path": str(path),
                "recorded_sha256": receipt["sha256"],
                "actual_sha256": actual,
                "matches": actual == receipt["sha256"],
                "original_readback_verified": receipt.get("readback_verified"),
            }
        )
    for name, expected in packet["files"].items():
        if name not in inputs.records and inputs.path(name).is_file():
            inputs.hash(name, "remaining frozen preparation artifact")
        if name not in inputs.records or inputs.records[name]["sha256"] != expected:
            issues.append(f"frozen preparation artifact missing or changed: {name}")
    inputs.verify_unchanged()
    return {
        "schema_version": 1,
        "analysis_only": True,
        "original_protocol_verdict": recorded.get("verdict"),
        "original_scientific_learning_confirmed": recorded.get("scientific_learning_confirmed"),
        "original_outcomes": recorded.get("outcomes"),
        "original_failure_stage": recorded.get("failure_stage"),
        "original_error": recorded.get("error"),
        "verdict_recomputed_or_changed": False,
        "protocol_id": packet.get("protocol_id"),
        "preparation_sha256": packet.get("sha256"),
        "evidence_consistency_issues": issues,
        "seeds": seeds,
        "archives": archive_checks,
        "recorded_provenance": {
            "sources": packet.get("sources"),
            "runtime": packet.get("runtime"),
            "data_files": manifest.get("data", {}).get("files"),
            "manifest_sha256": manifest.get("sha256"),
            "normalizer_sha256": normalizer.get("sha256"),
            "normalizer_sample_count": normalizer.get("sample_count"),
            "download_receipt_sha256": packet.get("download_receipt_sha256"),
        },
        "measurement_notes": [
            "Team diagnostics count environment transitions once, without agent-copy sums.",
            "Physical means need valid=1 and voltage_metrics_valid=1; failures remain attempted.",
            "Missing/nonfinite counts are explicit; incomplete metric means are null.",
            "Power sums are sums of sampled MW/MVAr values and are not energy integrals.",
            "Episode effects match exact episode ID, absolute start row, and environment seed.",
            "Training's generic abs(action)>.999 statistic is not physical-bound saturation.",
            "Source fingerprints come from preparation; live source is not rehashed.",
        ],
        "measurement_context": {
            "actor_observations": "zone-local features for each inverter's zone; not inverter-only",
            "units": {
                "voltage": "per_unit",
                "line_loss_power": "MW",
                "PV_power": "MW",
                "reactive_power": "MVAr",
                "action": "normalized dimensionless fraction",
            },
            "action_bounds": [-0.8, 0.8],
            "source_row_interval_seconds": manifest.get("data", {}).get("interval_seconds"),
            "first_timestamp_label": manifest.get("data", {}).get("first_timestamp"),
            "last_timestamp_label": manifest.get("data", {}).get("last_timestamp"),
            "timestamp_interpretation": "UTC labels; geographic timezone not established",
        },
        "inputs": inputs.records,
        "analysis_script_sha256": file_sha(__file__),
        "generated_utc": datetime.now(UTC).isoformat(),
        "analysis_python": sys.version,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.input_dir.resolve(), args.out_file.resolve()
    if output.is_relative_to(root):
        parser.error("--out-file must be outside the preserved evidence directory")
    if output.exists():
        parser.error("--out-file must be new; existing analysis is preserved")
    report = summarize(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
