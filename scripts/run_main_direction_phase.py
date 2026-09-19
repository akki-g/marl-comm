"""Prepare and execute the prospective bounded MAPDN main-direction phase.

Stages are separate so test windows remain unopened until all training closes.
Every scientific stage verifies the preparation and preserves failed attempts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from commstudy.analysis.diagnostics import finiteness_report, load_frozen_experiment
from commstudy.experiments.bookkeeping import RunContext, parameter_counts, read_json, utc_now
from commstudy.experiments.config import ExperimentSpec, resolved_experiment_dict
from commstudy.experiments.next_phase import (
    MAPDN_CALIBRATION_SEEDS,
    MAPDN_CHECKPOINTS,
    MAPDN_COMPARISON_SEEDS,
    MAPDN_HORIZON,
    MAPDN_METHODS,
    choose_mapdn_budget,
    content_sha256,
    select_nonoverlapping_rows,
)
from commstudy.experiments.runner import run_managed_experiment
from commstudy.tasks.torchrl.mapdn_data import (
    TrainingNormalizer,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from commstudy.tasks.torchrl.power_grids import TaskConfig
from run_mapdn_confirmation import (
    InitialCheckpointEvidence,
    _archive,
    _runtime,
    _spec,
    _validate_native_state,
    _write_new_json,
)
from run_mapdn_smoke import (
    _actor_parameters,
    _actor_state,
    _roll_bank,
    _sha,
    _source_provenance,
    _state_sha,
)


PROTOCOL_ID = "main_direction_mapdn_diagnostic_20260913_v1"
ROOT = Path(__file__).resolve().parents[1]


def source_inventory():
    names = [
        path
        for directory in ("src/commstudy", "configs", "scripts", "vendor")
        for path in (ROOT / directory).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
        and path.suffix != ".pyc"
    ]
    names.extend(
        ROOT / "docs" / name
        for name in ("MAIN_DIRECTION_NEXT_PHASE_PLAN.md", "PRIVATE_INFORMATION_DIAGNOSTIC_PLAN.md")
    )
    return {path.relative_to(ROOT).as_posix(): _sha(path) for path in sorted(names)}


def phase_spec(config, seed, method, frames, checkpoint_interval=8192):
    spec = _spec(ROOT, config, seed)
    model = json.loads(json.dumps(spec.model_config))
    params = model["params"]
    if method == "local_capacity":
        params.update(
            comm_class_path="commstudy.communication.local_control.LocalCapacityComm",
            comm_kwargs={"message_dim": 32, "grad_clip": None, "normalize_comm_path": False},
            comm_context_keys={},
        )
    elif method == "broadcast":
        params.update(
            comm_class_path="commstudy.communication.broadcast.BroadcastComm",
            comm_kwargs={
                "message_dim": 32,
                "exclude_self": True,
                "residual": True,
                "channel": {"type": "identity", "mode": "always"},
                "grad_clip": None,
                "normalize_comm_path": False,
            },
            comm_context_keys={"mask": "comm_mask", "sender_mask": "comm_sender_mask"},
        )
    elif method != "identity":
        raise ValueError("Undeclared phase method")
    spec = replace(
        spec,
        model=f"phase_{method}",
        model_config=model,
        experiment={
            **spec.experiment,
            "max_n_frames": frames,
            "checkpoint_interval": checkpoint_interval,
            "evaluation": False,
            "loggers": ["csv"],
        },
    )
    return ExperimentSpec(**resolved_experiment_dict(spec))


def prepare(output, data_path, previous_banks):
    before, runtime = source_inventory(), _runtime()
    output.mkdir(parents=True, exist_ok=False)
    old = read_json(previous_banks)
    old_sha = _sha(previous_banks)
    config = asdict(
        TaskConfig(
            data_path=str(data_path),
            manifest_path=str(output / "split_manifest.json"),
            max_steps=MAPDN_HORIZON,
            history=1,
            measurement_noise=False,
            reset_action=False,
        )
    )
    sources = _source_provenance(ROOT)
    manifest = build_split_manifest(
        data_path,
        MAPDN_HORIZON,
        history=1,
        source_revision=f"vendor-tree-sha256:{sources['mapdn_tree_sha256']}",
        settings={
            key: value
            for key, value in config.items()
            if key not in {"data_path", "manifest_path", "normalization_path"}
        },
    )
    save_split_manifest(manifest, output / "split_manifest.json")
    banks = {}
    exclusions = {
        split: [(row["start_row"], row["start_row"] + MAPDN_HORIZON + 1) for row in old[split]]
        for split in ("train", "validation", "test")
    }
    for name, split, count, seed in (
        ("normalization", "train", 16, 940000),
        ("calibration_validation", "validation", 8, 950000),
        ("comparison_validation", "validation", 8, 960000),
        ("comparison_test", "test", 16, 970000),
    ):
        rows = select_nonoverlapping_rows(
            manifest["splits"][split]["starts"],
            count,
            horizon=MAPDN_HORIZON,
            history=1,
            excluded=exclusions[split],
        )
        banks[name] = [
            {"episode_id": f"{PROTOCOL_ID}:{name}:{index}", "start_row": row, "seed": seed + index}
            for index, row in enumerate(rows)
        ]
        exclusions[split].extend((row, row + MAPDN_HORIZON + 1) for row in rows)
    _write_new_json(output / "banks.json", banks)
    _write_new_json(
        output / "precollection_declaration.json",
        {
            "created_utc": utc_now(),
            "protocol_id": PROTOCOL_ID,
            "source_files": before,
            "runtime": runtime,
            "banks_sha256": _sha(output / "banks.json"),
            "previous_banks_path": str(previous_banks),
            "previous_banks_sha256": old_sha,
            "purpose": "Declare rows and settings before any new normalization or evaluation",
            "test_rollouts_performed": 0,
        },
    )
    normalization, observations = _roll_bank(
        config, banks["normalization"], split="train", retain_observations=True
    )
    _write_new_json(output / "normalization_collection.json", normalization)
    if normalization["solver_failure_count"] or any(
        row["transition_count"] != MAPDN_HORIZON for row in normalization["episodes"]
    ):
        raise ValueError("Normalization bank did not complete every healthy training episode")
    normalizer = TrainingNormalizer.fit(
        np.stack(observations), split="train", manifest_sha256=manifest["sha256"]
    )
    normalizer.save(output / "normalizer.json")
    config["normalization_path"] = str(output / "normalizer.json")
    _write_new_json(output / "task_config.json", config)
    from commstudy.analysis.mapdn_paired import build_mapdn_bank

    for name, split in (
        ("calibration_validation", "validation"),
        ("comparison_validation", "validation"),
        ("comparison_test", "test"),
    ):
        contract = build_mapdn_bank(
            config, banks[name], split=split, purpose=name, source_root=ROOT
        )
        _write_new_json(output / f"{name}_contract.json", contract)
    replay_bank = build_mapdn_bank(
        config,
        banks["comparison_validation"][:2],
        split="validation",
        purpose="first_two_comparison_validation_exact_replay",
        source_root=ROOT,
    )
    _write_new_json(output / "comparison_replay_contract.json", replay_bank)
    specs = output / "specs"
    specs.mkdir()
    for seed in MAPDN_CALIBRATION_SEEDS:
        _write_new_json(
            specs / f"calibration_identity_{seed}.json",
            asdict(phase_spec(config, seed, "identity", MAPDN_CHECKPOINTS[-1])),
        )
    # Freeze all allowed specifications now; selection later only chooses their budget.
    for frames in MAPDN_CHECKPOINTS[2:]:
        for seed in MAPDN_COMPARISON_SEEDS:
            for method in MAPDN_METHODS:
                _write_new_json(
                    specs / f"comparison_{method}_{seed}_{frames}.json",
                    asdict(phase_spec(config, seed, method, frames)),
                )
    if before != source_inventory() or runtime != _runtime() or old_sha != _sha(previous_banks):
        raise ValueError("Source/runtime/previous bank changed during preparation")
    files = {
        path.relative_to(output).as_posix(): _sha(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    packet = {
        "protocol_id": PROTOCOL_ID,
        "prepared_utc": utc_now(),
        "root": str(ROOT),
        "output": str(output),
        "data_path": str(data_path),
        "source_files": before,
        "runtime": runtime,
        "files": files,
        "test_rollouts_performed": 0,
        "scientific_learning_confirmed": False,
    }
    packet["sha256"] = content_sha256(packet)
    _write_new_json(output / "preparation.json", packet)
    return packet


def verify(output, expected_sha):
    packet = read_json(output / "preparation.json")
    digest = packet.pop("sha256")
    if digest != expected_sha or digest != content_sha256(packet):
        raise ValueError("Preparation content hash differs")
    packet["sha256"] = digest
    if packet["protocol_id"] != PROTOCOL_ID or packet["root"] != str(ROOT):
        raise ValueError("Preparation identity/source path differs")
    if packet["output"] != str(output) or packet["source_files"] != source_inventory():
        raise ValueError("Prepared source or output path differs")
    if packet["runtime"] != _runtime():
        raise ValueError("Prepared scientific runtime differs")
    for name, sha in packet["files"].items():
        if _sha(output / name) != sha:
            raise ValueError(f"Prepared artifact changed: {name}")
    load_split_manifest(output / "split_manifest.json", packet["data_path"])
    return packet


def _calibration_point(summary, frames):
    means = summary["domain_means"]
    return {
        "frames": frames,
        "return": summary["mean_episode_return"],
        "voltage_magnitude": means["voltage_violation_magnitude"],
        "violation_frequency": summary["violated_bus_fraction_observed"],
        "q_effort": means["q_loss"],
        "solver_failures": summary["solver_failure_count"],
        "complete": summary["full_horizon_episodes"] == summary["episode_count"]
        and summary["solver_failure_count"] == 0,
    }


def action_diagnostics(episodes):
    actions = np.asarray(
        [step["actions"] for episode in episodes for step in episode["transitions"]]
    )
    if not np.isfinite(actions).all() or actions.ndim != 3 or actions.shape[-1] != 1:
        raise ValueError("Invalid saved MAPDN action array")
    absolute = np.abs(actions)
    saturated = absolute / 0.8 > 0.999
    return {
        "physical_bounds": [-0.8, 0.8],
        "saturation_rule": "abs(action)/0.8 > 0.999",
        "action_scalar_count": int(actions.size),
        "saturated_scalar_count": int(saturated.sum()),
        "saturation_fraction": float(saturated.mean()),
        "mean_abs_action": float(absolute.mean()),
        "max_abs_action": float(absolute.max()),
    }


class PhaseEvidence(InitialCheckpointEvidence):
    def __init__(self, config, out_dir, preparation_sha, bank, checkpoints):
        super().__init__(config, out_dir, preparation_sha)
        self.bank = bank
        self.checkpoints = checkpoints
        self.points = []

    def measure(self, frames):
        from commstudy.analysis.mapdn_paired import evaluate_mapdn_paired

        result = evaluate_mapdn_paired(self.experiment, self.bank, arms=("live",))
        result["action_diagnostics"] = action_diagnostics(result["arms"]["live"])
        _write_new_json(self.out_dir / f"validation_{frames}.json", result)
        self.points.append(_calibration_point(result["summaries"]["live"], frames))

    def on_setup(self):
        super().on_setup()
        self.measure(0)

    def on_train_end(self, training_td, group):
        if group != "agents":
            raise ValueError("Unexpected MAPDN optimization group")
        frames = self.experiment.total_frames
        if frames in self.checkpoints:
            self.measure(frames)


def train_one(output, packet, stage, method, seed):
    if stage == "calibration":
        if seed not in MAPDN_CALIBRATION_SEEDS or method != "identity":
            raise ValueError("Only the declared Identity calibration seeds may run")
        frames = MAPDN_CHECKPOINTS[-1]
        spec_name = f"calibration_identity_{seed}.json"
        schedule = MAPDN_CHECKPOINTS
        bank_name = "calibration_validation"
    else:
        decision = read_json(output / "budget_decision.json")
        validate_budget_decision(output, packet, decision)
        frames = decision["decision"]["selected_frames"]
        if seed not in MAPDN_COMPARISON_SEEDS or method not in MAPDN_METHODS:
            raise ValueError("Undeclared comparison allocation")
        spec_name = f"comparison_{method}_{seed}_{frames}.json"
        schedule = (0, frames)
        bank_name = "comparison_validation"
    spec = ExperimentSpec(**read_json(output / "specs" / spec_name))
    evidence = output / "evidence" / stage / f"{method}_{seed}"
    evidence.mkdir(parents=True, exist_ok=False)
    callback = PhaseEvidence(
        spec.task_config["params"],
        evidence,
        packet["sha256"],
        read_json(output / f"{bank_name}_contract.json"),
        schedule,
    )
    context = RunContext(
        suite_id=f"{PROTOCOL_ID}_{stage}",
        run_id=f"{method}_{seed}",
        output_root=output / "runs",
        command=tuple(sys.argv),
    )
    started = time.monotonic()
    experiment = run_managed_experiment(spec, context, repo_root=ROOT, callbacks=[callback])
    try:
        final_parameters = _actor_parameters(experiment)
        checks = {
            "exact_frames": callback.collected_frames == frames,
            "exact_batches": callback.batch_counts == [256] * (frames // 256),
            "actor_changed": any(
                not torch.equal(callback.initial_parameters[name], value)
                for name, value in final_parameters.items()
            ),
            "actor_finite": all(
                torch.isfinite(value).all().item() for value in final_parameters.values()
            ),
            "validation_schedule": tuple(point["frames"] for point in callback.points)
            == tuple(schedule),
            "initial_actor_unchanged": _sha(evidence / "initial_actor.pt")
            == callback.initial_manifest["file_sha256"],
        }
        native = {}
        for checkpoint_frames in range(8192, frames + 1, 8192):
            paths = list(
                context.run_dir.glob(f"benchmarl/**/checkpoints/checkpoint_{checkpoint_frames}.pt")
            )
            if len(paths) != 1:
                raise ValueError("Missing or duplicated native checkpoint")
            saved = torch.load(paths[0], weights_only=False, map_location="cpu")
            _validate_native_state(
                saved,
                experiment.losses,
                checkpoint_frames,
                require_equal=checkpoint_frames == frames,
            )
            native[str(checkpoint_frames)] = {"path": str(paths[0]), "sha256": _sha(paths[0])}
        health = finiteness_report(context.run_dir)
        checks["logged_metrics_finite"] = health["all_finite"] and health["total_values"] > 0
        result = {
            "stage": stage,
            "method": method,
            "seed": seed,
            "frames": frames,
            "finished_utc": utc_now(),
            "preparation_sha256": packet["sha256"],
            "integrity_passed": all(checks.values()),
            "checks": checks,
            "validation": callback.points,
            "health": health,
            "run_dir": str(context.run_dir),
            "evidence_dir": str(evidence),
            "native_checkpoints": native,
            "initial_actor": callback.initial_manifest,
            "parameter_counts": parameter_counts(experiment),
            "final_actor_state_sha256": _state_sha(_actor_state(experiment)),
            "elapsed_seconds": time.monotonic() - started,
            "test_rollouts_performed": 0,
        }
    finally:
        experiment.close()
    if stage == "comparison":
        from commstudy.analysis.mapdn_paired import (
            evaluate_mapdn_paired,
            verify_native_checkpoint,
        )

        reloaded = load_frozen_experiment(context.run_dir, scratch_root=evidence / "reload_scratch")
        try:
            if _state_sha(_actor_state(reloaded)) != result["final_actor_state_sha256"]:
                raise ValueError("Strictly reloaded actor differs from completed training")
            result["native_reload"] = verify_native_checkpoint(
                reloaded, native[str(frames)]["path"], expected_frames=frames
            )
            bank = read_json(output / "comparison_replay_contract.json")
            first = evaluate_mapdn_paired(
                reloaded, bank, require_exact_local_null=method != "broadcast"
            )
            second = evaluate_mapdn_paired(
                reloaded, bank, require_exact_local_null=method != "broadcast"
            )
            _write_new_json(evidence / "paired_validation.json", first)
            _write_new_json(evidence / "paired_validation_replay.json", second)
            result["checks"]["exact_validation_replay"] = first == second
            result["validation_replay_sha256"] = _sha(evidence / "paired_validation.json")
            result["integrity_passed"] = all(result["checks"].values())
        finally:
            reloaded.close()
    verify(output, packet["sha256"])
    archives = output / "archives"
    archives.mkdir(exist_ok=True)
    result["run_archive"] = _archive(context.run_dir, archives / f"{stage}_{method}_{seed}.tar.gz")
    _write_new_json(evidence / "training_summary.json", result)
    if not result["integrity_passed"]:
        raise ValueError("Training integrity failed; retained row cannot authorize expansion")
    return result


def calibration_summaries(output):
    return [
        output / "evidence" / "calibration" / f"identity_{seed}" / "training_summary.json"
        for seed in MAPDN_CALIBRATION_SEEDS
    ]


def select_budget(output, packet):
    paths = calibration_summaries(output)
    decision = choose_mapdn_budget([read_json(path) for path in paths])
    result = {
        "created_utc": utc_now(),
        "preparation_sha256": packet["sha256"],
        "decision": decision,
        "inputs": {str(path): _sha(path) for path in paths},
        "comparison_or_test_outcomes_used": False,
    }
    _write_new_json(output / "budget_decision.json", result)
    return result


def validate_budget_decision(output, packet, decision):
    if decision["preparation_sha256"] != packet["sha256"]:
        raise ValueError("Budget decision belongs to another preparation")
    paths = calibration_summaries(output)
    if decision["inputs"] != {str(path): _sha(path) for path in paths}:
        raise ValueError("Baseline input evidence changed after allocation")
    if decision["decision"] != choose_mapdn_budget([read_json(path) for path in paths]):
        raise ValueError("Budget selection differs from method-blind declared rule")


def close_training(output, packet):
    decision = read_json(output / "budget_decision.json")
    validate_budget_decision(output, packet, decision)
    rows = []
    for seed in MAPDN_COMPARISON_SEEDS:
        for method in MAPDN_METHODS:
            path = output / "evidence" / "comparison" / f"{method}_{seed}" / "training_summary.json"
            row = read_json(path)
            if (
                row["preparation_sha256"] != packet["sha256"]
                or not row["integrity_passed"]
                or row["frames"] != decision["decision"]["selected_frames"]
                or row["seed"] != seed
                or row["method"] != method
            ):
                raise ValueError("Comparison cohort is incomplete or differs from allocation")
            metadata = read_json(Path(row["run_dir"]) / "metadata.json")
            policy = Path(row["run_dir"]) / metadata["policy_checkpoint"]
            native = row["native_checkpoints"][str(row["frames"])]
            if _sha(native["path"]) != native["sha256"]:
                raise ValueError("Retained final native checkpoint changed")
            if not row["run_archive"]["readback_verified"] or (
                _sha(row["run_archive"]["path"]) != row["run_archive"]["sha256"]
            ):
                raise ValueError("Verified pretest managed-run archive changed")
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "training_summary_path": str(path),
                    "training_summary_sha256": _sha(path),
                    "policy_path": str(policy),
                    "policy_sha256": _sha(policy),
                    "native_checkpoint": native,
                }
            )
    result = {
        "closed_utc": utc_now(),
        "preparation_sha256": packet["sha256"],
        "budget_decision_sha256": _sha(output / "budget_decision.json"),
        "complete_policies": len(rows),
        "rows": rows,
        "test_rollouts_performed": 0,
    }
    _write_new_json(output / "comparison_training_complete.json", result)
    return result


def evaluate_one(output, packet, method, seed):
    from commstudy.analysis.mapdn_paired import evaluate_mapdn_paired, verify_native_checkpoint

    closure = read_json(output / "comparison_training_complete.json")
    if closure["preparation_sha256"] != packet["sha256"] or closure["complete_policies"] != 15:
        raise ValueError("All fifteen final training policies must close before test access")
    decision = read_json(output / "budget_decision.json")
    validate_budget_decision(output, packet, decision)
    if closure["budget_decision_sha256"] != _sha(output / "budget_decision.json"):
        raise ValueError("Budget changed after pretest training closure")
    expected_roster = {(s, m) for s in MAPDN_COMPARISON_SEEDS for m in MAPDN_METHODS}
    if (
        len(closure["rows"]) != 15
        or {(row["seed"], row["method"]) for row in closure["rows"]} != expected_roster
    ):
        raise ValueError("Closed training roster is incomplete or duplicated")
    for row in closure["rows"]:
        if _sha(row["training_summary_path"]) != row["training_summary_sha256"] or (
            _sha(row["policy_path"]) != row["policy_sha256"]
        ):
            raise ValueError("Closed training/checkpoint evidence changed")
        summary = read_json(row["training_summary_path"])
        native = row["native_checkpoint"]
        archive = summary["run_archive"]
        if (
            _sha(native["path"]) != native["sha256"]
            or not archive["readback_verified"]
            or (_sha(archive["path"]) != archive["sha256"])
        ):
            raise ValueError("Closed native checkpoint or verified training archive changed")
    selected = [row for row in closure["rows"] if row["seed"] == seed and row["method"] == method]
    if len(selected) != 1:
        raise ValueError("Undeclared test row")
    row = read_json(selected[0]["training_summary_path"])
    destination = output / "evaluations" / f"{method}_{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    experiment = load_frozen_experiment(
        Path(row["run_dir"]), scratch_root=destination / "reload_scratch"
    )
    try:
        if _state_sha(_actor_state(experiment)) != row["final_actor_state_sha256"]:
            raise ValueError("Reloaded policy differs from closed final actor")
        native_proof = verify_native_checkpoint(
            experiment,
            row["native_checkpoints"][str(row["frames"])]["path"],
            expected_frames=row["frames"],
        )
        bank = read_json(output / "comparison_test_contract.json")
        result = evaluate_mapdn_paired(
            experiment,
            bank,
            arms=("live", "severed"),
            require_exact_local_null=method != "broadcast",
        )
        result.update(
            seed=seed,
            method=method,
            training_summary_sha256=_sha(selected[0]["training_summary_path"]),
            preparation_sha256=packet["sha256"],
            native_checkpoint_verification=native_proof,
        )
        result["action_diagnostics"] = {
            arm: action_diagnostics(episodes) for arm, episodes in result["arms"].items()
        }
        _write_new_json(destination / "paired_test.json", result)
    finally:
        experiment.close()
    verify(output, packet["sha256"])
    return {
        "path": str(destination / "paired_test.json"),
        "sha256": _sha(destination / "paired_test.json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "calibrate",
            "select-budget",
            "train",
            "close-training",
            "evaluate",
            "verify",
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--previous-banks", type=Path)
    parser.add_argument("--preparation-sha256")
    parser.add_argument("--method", choices=MAPDN_METHODS, default="identity")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    output = args.out_dir.resolve()
    if args.stage != "prepare" and not args.preparation_sha256:
        parser.error("Scientific stages require --preparation-sha256")
    if args.stage == "prepare" and (args.data_path is None or args.previous_banks is None):
        parser.error("Preparation requires real --data-path and explicit --previous-banks")
    try:
        if args.stage == "prepare":
            result = prepare(output, args.data_path.resolve(), args.previous_banks.resolve())
        else:
            packet = verify(output, args.preparation_sha256)
            if args.stage in ("calibrate", "train"):
                result = train_one(
                    output,
                    packet,
                    "calibration" if args.stage == "calibrate" else "comparison",
                    args.method,
                    args.seed,
                )
            elif args.stage == "select-budget":
                result = select_budget(output, packet)
            elif args.stage == "close-training":
                result = close_training(output, packet)
            elif args.stage == "evaluate":
                result = evaluate_one(output, packet, args.method, args.seed)
            else:
                result = {"verified": True, "preparation_sha256": packet["sha256"]}
        # The external supervisor owns durable stdout; no background process depends on this pipe.
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "out_dir": str(output),
                    "result_sha256": content_sha256(result),
                },
                allow_nan=False,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        if output.exists():
            failures = output / "failures"
            failures.mkdir(exist_ok=True)
            name = f"{time.time_ns()}_{args.stage}_{args.method}_{args.seed}.json"
            _write_new_json(
                failures / name,
                {
                    "failed_utc": utc_now(),
                    "stage": args.stage,
                    "seed": args.seed,
                    "method": args.method,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
