"""Real training, strict reconstruction, closure and failure-preserving execution."""

from __future__ import annotations
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import torch
from benchmarl.experiment.callback import Callback

from commstudy.experiments.bookkeeping import (
    RunContext,
    parameter_counts,
    read_json,
    scheduler_identity,
    utc_now,
)
from commstudy.experiments.mechanism_suite import (
    ROOT,
    SUITE,
    CONDITIONS,
    CHECKPOINTS,
    contract_key,
    effective_spec,
    exclusive,
    file_sha,
    gate,
    inventory,
    latest_attempt,
    manifest,
    runtime,
    select_row,
    validate_launch,
    verify_approval,
    verify_plan,
    verify_preflight,
    verify_receipt,
    write_new,
)


def preflight(root, task, profile, *, local=False):
    from commstudy.experiments.protocols import expected_model_contract
    from commstudy.experiments.mechanism_data import historical_inventory, prepare_mapdn
    from commstudy.experiments.mechanism_slurm import load_profile

    root = Path(root)
    packet = verify_plan(root)
    profile_doc = load_profile(profile, root)
    if not local and not os.getenv("SLURM_JOB_ID"):
        raise ValueError(
            "Real-task preflight must run in an allocation; --local-check labels local evidence"
        )
    if not local:
        probe = gate(root, "cluster_probe")
        if probe["profile_sha256"] != file_sha(profile):
            raise ValueError("Cluster probe does not bind this runtime/resource profile")
    if profile_doc["SOURCE_ROOT"] != str(ROOT):
        raise ValueError("Profile source root differs from imports")
    interpreter = profile_doc[f"{task.upper()}_PYTHON"]
    if Path(sys.executable).absolute() != Path(interpreter).absolute():
        raise ValueError("Wrong task interpreter; source the isolated profile")
    target = root / "gates" / f"preflight_{task}.json"
    if target.exists():
        return verify_preflight(root, task)
    with exclusive(root / f".preflight_{task}.lock"):
        current_runtime = runtime(task)
        free = __import__("shutil").disk_usage(root).free
        minimum = int(profile_doc["MIN_FREE_GIB"]) * 1024**3
        if free < minimum:
            raise ValueError(f"Insufficient storage: {free} bytes free; profile needs {minimum}")
        doc = read_json(root / "preparation/suite.json")
        history = historical_inventory(doc)
        write_new(root / f"preparation/history_{task}.json", history)
        if task == "mapdn":
            prepare_mapdn(root, packet, history)
        contracts = {}
        for stage in ("smoke", "qualification", "main"):
            for row in manifest(root, task, stage):
                key = contract_key(row)
                if key not in contracts:
                    contracts[key] = expected_model_contract(effective_spec(root, row))
        write_new(root / f"preparation/contracts_{task}.json", contracts)
        write_new(root / f"preparation/runtime_{task}.json", current_runtime)
        # Each task binds only its own materialization, avoiding races across preflights.
        names = [f"preparation/{n}_{task}.json" for n in ("history", "contracts", "runtime")]
        if task == "mapdn":
            names += [
                "preparation/" + n
                for n in (
                    "split_manifest.json",
                    "normalizer.json",
                    "normalization_collection.json",
                    "data_inventory.json",
                    "mapdn_banks.json",
                    "mapdn_qualification_bank.json",
                    "mapdn_validation_bank.json",
                    "mapdn_test_bank.json",
                )
            ]
            if "mapdn_topology" in doc["include"]:
                names.append("preparation/topologies.json")
        inputs = {n: file_sha(root / n) for n in names}
        if not local:
            inputs["gates/cluster_probe.json"] = file_sha(root / "gates/cluster_probe.json")
        result = {
            "status": "passed",
            "task": task,
            "plan_sha256": packet["sha256"],
            "source_sha256": packet["source_sha256"],
            "runtime": current_runtime,
            "inputs": inputs,
            "profile_path": str(Path(profile).resolve()),
            "profile_sha256": file_sha(profile),
            "cluster_profile_verified": not local,
            "scheduler": scheduler_identity(),
            "free_bytes": free,
            "cost_estimates": "unmeasured; use real smoke and qualification receipts",
            "test_rollouts_performed": 0,
        }
        if inventory() != read_json(root / "preparation/source_inventory.json"):
            raise ValueError("Source changed during preflight")
        write_new(target, result)
        return result


def evaluate_development(experiment, root, row, *, small=False, random_actions=False):
    if row["task"] == "pcp":
        from commstudy.analysis.mechanism_suite import evaluate_pcp

        doc = read_json(root / "preparation/suite.json")
        seeds = doc["pcp_random_audit_seeds" if random_actions else "pcp_development_seeds"]
        return evaluate_pcp(
            experiment,
            seeds[:2] if small else seeds,
            local_null=row["method"].startswith(("identity", "local_capacity")),
            random_actions=random_actions,
        )
    from commstudy.analysis.mapdn_paired import build_mapdn_bank, evaluate_mapdn_paired

    name = "qualification" if row["stage"] == "qualification" else "validation"
    bank = read_json(root / f"preparation/mapdn_{name}_bank.json")
    if small:
        entries = [
            {k: entry[k] for k in ("episode_id", "seed", "start_row")}
            for entry in bank["episodes"][:2]
        ]
        bank = build_mapdn_bank(
            dict(experiment.task.config),
            entries,
            split="validation",
            purpose="smoke_only",
            source_root=ROOT,
        )
    return evaluate_mapdn_paired(
        experiment,
        bank,
        require_exact_local_null=row["method"].startswith(("identity", "local_capacity")),
    )


class Evidence(Callback):
    def __init__(self, root, row, output):
        self.root, self.row, self.output = root, row, output
        self.collected = 0
        self.batches = []
        self.points = []
        self.measure_seconds = 0.0
        self.schedule = CHECKPOINTS[row["task"]] if row["stage"] != "smoke" else [row["frames"]]

    def measure(self, frame):
        start = time.monotonic()
        value = evaluate_development(
            self.experiment, self.root, self.row, small=self.row["stage"] == "smoke"
        )
        write_new(self.output / f"validation_{frame}.json", value)
        self.points.append({"frames": frame, "summaries": value["summaries"]})
        self.measure_seconds += time.monotonic() - start

    def on_setup(self):
        group = "adversary" if self.row["task"] == "pcp" else "agents"
        self.initial = {
            k: p.detach().cpu().clone()
            for k, p in self.experiment.group_policies[group].named_parameters()
        }
        self.measure(0)

    def on_batch_collected(self, batch):
        self.collected += batch.numel()
        self.batches.append(batch.numel())

    def on_train_end(self, training_td, group):
        target = "adversary" if self.row["task"] == "pcp" else "agents"
        frame = self.experiment.total_frames
        if (
            group == target
            and frame in self.schedule
            and frame not in [p["frames"] for p in self.points]
        ):
            self.measure(frame)


def failure_category(exc):
    from commstudy.experiments.health import TrainingHealthError

    if isinstance(exc, (TimeoutError, InterruptedError, KeyboardInterrupt, OSError, MemoryError)):
        return "infrastructure_failure"
    if isinstance(exc, FloatingPointError) and (
        not isinstance(exc, TrainingHealthError) or "replay" not in str(exc).lower()
    ):
        return "scientific_failure"
    return "integrity_failure"


def allocate_attempt(root, row, kind="runs", retry=None):
    base = root / kind / row["task"] / row["run_id"]
    attempts = sorted(base.glob("attempt_*"))
    if attempts:
        prior = latest_attempt(root, row, kind)
        if not retry:
            raise ValueError("Existing attempt preserved; explicit retry plan required")
        approval = read_json(retry)
        if (
            approval.get("plan_sha256") != verify_plan(root)["sha256"]
            or approval.get("kind") != kind
            or row["index"] not in approval.get("indices", [])
            or approval.get("stage") != row["stage"]
            or approval.get("task") != row["task"]
        ):
            raise ValueError("Retry authorization does not bind this row/stage")
        if not prior or prior["status"] != "infrastructure_failure":
            raise ValueError("Only recorded infrastructure failures may be retried")
        receipt = attempts[-1] / "receipt.json"
        if approval["prior_receipts"].get(str(row["index"])) != file_sha(receipt):
            raise ValueError("Retry plan is stale or already consumed")
    elif retry:
        raise ValueError("Retry cannot launch an unattempted row")
    output = base / f"attempt_{len(attempts):03d}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _interrupt(signum, _frame):
    raise InterruptedError(f"Received signal {signum}; checkpoint-only exact resume is unsupported")


def train(root, task, stage, index, *, retry=None):
    from commstudy.experiments.runner import run_managed_experiment
    from commstudy.analysis.diagnostics import finiteness_report
    from commstudy.analysis.mechanism_suite import verify_suite_checkpoint
    from run_mapdn_confirmation import _validate_native_state

    root = Path(root)
    row = select_row(root, task, stage, index)
    binding = {"kind": SUITE, "run_root": str(root), "task": task, "stage": stage, "index": index}
    spec = effective_spec(root, row)
    validate_launch(spec, binding, repo_root=ROOT)
    if (root / f"gates/training_closure_{task}.json").exists() and stage == "main":
        raise ValueError("Training is sealed; no attempts after closure")
    with exclusive(root / "locks" / task / row["run_id"]):
        output = allocate_attempt(root, row, retry=retry)
        start = time.monotonic()
        packet = verify_plan(root)
        start_record = {
            "row": row,
            "scheduler": scheduler_identity(),
            "pid": os.getpid(),
            "plan_sha256": packet["sha256"],
            "runtime": runtime(task),
            "source_sha256": packet["source_sha256"],
            "created_utc": utc_now(),
        }
        write_new(output / "start.json", start_record)
        context = RunContext(
            suite_id="managed",
            run_id=row["run_id"],
            output_root=output,
            command=tuple(sys.argv),
            attempt=int(output.name[-3:]),
        )
        callback = Evidence(root, row, output)
        contracts = read_json(root / f"preparation/contracts_{task}.json")
        experiment = frozen = None
        old = {s: signal.signal(s, _interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            experiment = run_managed_experiment(
                spec,
                context,
                repo_root=ROOT,
                callbacks=[callback],
                protocol_binding=binding,
                expected_contract=contracts[contract_key(row)],
            )
            group = "adversary" if task == "pcp" else "agents"
            current = dict(experiment.group_policies[group].named_parameters())
            checks = {
                "exact_frames": callback.collected == row["frames"],
                "exact_batches": callback.batches
                == [6000 if task == "pcp" else 256]
                * (row["frames"] // (6000 if task == "pcp" else 256)),
                "actor_changed": any(
                    not torch.equal(callback.initial[k], p.detach().cpu())
                    for k, p in current.items()
                ),
                "actor_finite": all(torch.isfinite(p).all().item() for p in current.values()),
                "validation_schedule": [p["frames"] for p in callback.points]
                == [0, *callback.schedule],
            }
            health = finiteness_report(context.run_dir)
            checks["logged_finite"] = health["all_finite"] and health["total_values"] > 0
            native = {}
            for frame in callback.schedule:
                paths = list(context.run_dir.glob(f"benchmarl/*/checkpoints/checkpoint_{frame}.pt"))
                if len(paths) != 1:
                    raise ValueError(
                        f"Mandatory checkpoint missing/duplicate after cleanup: {frame}"
                    )
                saved = torch.load(paths[0], weights_only=False, map_location="cpu")
                _validate_native_state(
                    saved, experiment.losses, frame, require_equal=frame == row["frames"]
                )
                native[str(frame)] = {"path": str(paths[0]), "sha256": file_sha(paths[0])}
            counts = parameter_counts(experiment)
            for group_name, policy in experiment.group_policies.items():
                for component, fragment in (("encoder", "encoders"), ("head", "output_heads")):
                    counts[f"group/{group_name}/{component}_trainable"] = sum(
                        p.numel()
                        for name, p in policy.named_parameters()
                        if fragment in name and p.requires_grad
                    )
            experiment.close()
            experiment = None
            # Exercise the same serialized string route used after training closure.
            summary = {
                "run_dir": str(context.run_dir),
                "native_checkpoints": native,
                "frames": row["frames"],
            }
            write_new(output / "training_summary.json", summary)
            frozen = reload_closed_policy(
                str(output / "training_summary.json"), output / "reload_scratch"
            )
            final_native = native[str(row["frames"])]["path"]
            parity = verify_suite_checkpoint(frozen, final_native, expected_frames=row["frames"])
            development = evaluate_development(frozen, root, row, small=stage == "smoke")
            repeat = evaluate_development(frozen, root, row, small=stage == "smoke")
            checks["exact_development_replay"] = development == repeat
            write_new(output / "paired_development.json", development)
            if task == "pcp" and stage == "smoke":
                write_new(
                    output / "random_action_audit.json",
                    evaluate_development(frozen, root, row, small=True, random_actions=True),
                )
            if not all(checks.values()):
                raise ValueError(f"Training integrity check failed: {checks}")
            verify_preflight(root, task)
            from run_mapdn_confirmation import _archive

            archive_started = time.monotonic()
            archive = _archive(context.run_dir, output / "managed_run.tar.gz")
            write_new(output / "archive_readback.json", archive)
            archive_seconds = time.monotonic() - archive_started
            paths = [
                output / "training_summary.json",
                output / "paired_development.json",
                context.run_dir / "metadata.json",
                context.run_dir / "resolved_config.yaml",
                context.run_dir / "checkpoints/policy_state.pt",
                *[Path(v["path"]) for v in native.values()],
                *output.glob("validation_*.json"),
            ]
            paths += list(context.run_dir.glob("metrics*.csv"))
            paths += [output / "managed_run.tar.gz", output / "archive_readback.json"]
            artifacts = {p.relative_to(root).as_posix(): file_sha(p) for p in paths}
            record = {
                "status": "completed",
                "integrity_passed": True,
                "row": row,
                "plan_sha256": packet["sha256"],
                "checks": checks,
                "native_parity": parity,
                "training_summary_path": str(output / "training_summary.json"),
                "artifacts": artifacts,
                "parameter_counts": counts,
                "validation": callback.points,
                "elapsed_seconds": time.monotonic() - start,
                "development_evaluation_seconds": callback.measure_seconds,
                "archive_seconds": archive_seconds,
                "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_rss_units": "bytes" if sys.platform == "darwin" else "KiB",
                "retained_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                "retained_files": sum(p.is_file() for p in output.rglob("*")),
                "budget_adequacy": "unresolved; fixed-budget curve retained",
                "receipt_path": (output / "receipt.json").relative_to(root).as_posix(),
            }
            write_new(output / "receipt.json", record)
            return record
        except BaseException as exc:
            checkpoints = {
                str(p): file_sha(p) for p in output.glob("managed/*/benchmarl/*/checkpoints/*.pt")
            }
            write_new(
                output / "receipt.json",
                {
                    "status": failure_category(exc),
                    "row": row,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "phase": stage,
                    "actual_frames": callback.collected,
                    "available_checkpoints": checkpoints,
                    "plan_sha256": packet["sha256"],
                    "source_sha256": packet["source_sha256"],
                    "runtime": start_record["runtime"],
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
            raise
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
            if experiment is not None:
                experiment.close()
            if frozen is not None:
                frozen.close()


def reload_closed_policy(summary_path, scratch):
    from commstudy.analysis.diagnostics import load_frozen_experiment

    summary = read_json(summary_path)
    if not summary or not isinstance(summary.get("run_dir"), str):
        raise ValueError("Training summary must serialize its run directory")
    return load_frozen_experiment(summary["run_dir"], scratch_root=scratch)


def close_training(root, task):
    root = Path(root)
    verify_approval(root)
    rows, artifacts = [], {}
    for row in manifest(root, task):
        result = latest_attempt(root, row)
        if not result or result["status"] not in {"completed", "scientific_failure"}:
            raise ValueError(f"Unresolved training row blocks closure: {row['run_id']}")
        if (root / "locks" / task / row["run_id"]).exists():
            raise ValueError("A worker still owns the row")
        verify_receipt(root, result)
        receipt = sorted((root / "runs" / task / row["run_id"]).glob("attempt_*/receipt.json"))[-1]
        artifacts[receipt.relative_to(root).as_posix()] = file_sha(receipt)
        artifacts.update(result.get("artifacts", {}))
        rows.append(
            {
                "index": row["index"],
                "run_id": row["run_id"],
                "status": result["status"],
                "receipt_path": str(receipt),
            }
        )
    closure = {
        "status": "passed",
        "rows": rows,
        "artifacts": artifacts,
        "plan_sha256": verify_plan(root)["sha256"],
        "created_utc": utc_now(),
        "cohort_complete": all(r["status"] == "completed" for r in rows),
    }
    write_new(root / f"gates/training_closure_{task}.json", closure)
    return closure


def evaluate(root, task, index, *, retry=None):
    from commstudy.analysis.mapdn_paired import evaluate_mapdn_paired
    from commstudy.analysis.mechanism_suite import verify_suite_checkpoint
    from commstudy.analysis.mechanism_suite import evaluate_pcp

    root = Path(root)
    verify_approval(root)
    verify_preflight(root, task)
    # Both core tasks must seal training/settings before any final test is opened.
    for name in CONDITIONS:
        verify_receipt(root, gate(root, f"training_closure_{name}"))
    row = select_row(root, task, "main", index)
    trained = latest_attempt(root, row)
    with exclusive(root / "locks" / task / f"eval_{row['run_id']}"):
        output = allocate_attempt(root, row, "evaluation", retry=retry)
        write_new(
            output / "start.json",
            {
                "scheduler": scheduler_identity(),
                "row": row,
                "plan_sha256": verify_plan(root)["sha256"],
                "created_utc": utc_now(),
            },
        )
        if trained["status"] == "scientific_failure":
            result = {
                "status": "not_evaluable",
                "reason": "preserved scientific training failure",
                "row": row,
            }
            write_new(output / "receipt.json", result)
            return result
        experiment = None
        started = time.monotonic()
        old = {s: signal.signal(s, _interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            verify_receipt(root, trained)
            experiment = reload_closed_policy(trained["training_summary_path"], output / "scratch")
            summary = read_json(trained["training_summary_path"])
            native = summary["native_checkpoints"][str(row["frames"])]
            parity = verify_suite_checkpoint(
                experiment, native["path"], expected_frames=row["frames"]
            )
            local = row["method"].startswith(("identity", "local_capacity"))
            if task == "pcp":
                seeds = read_json(root / "preparation/suite.json")["pcp_test_seeds"]

                def measure():
                    return evaluate_pcp(experiment, seeds, local_null=local)
            else:
                bank = read_json(root / "preparation/mapdn_test_bank.json")

                def measure():
                    return evaluate_mapdn_paired(experiment, bank, require_exact_local_null=local)

            result = measure()
            write_new(output / "evaluation.json", result)
            replay = measure()
            write_new(output / "replay.json", replay)
            if result != replay:
                raise ValueError("Full paired evaluation did not replay exactly")
            verify_receipt(root, trained)
            artifacts = {
                p.relative_to(root).as_posix(): file_sha(p)
                for p in (output / "evaluation.json", output / "replay.json")
            }
            receipt = {
                "status": "completed",
                "row": row,
                "artifacts": artifacts,
                "plan_sha256": verify_plan(root)["sha256"],
                "native_parity": parity,
                "result_path": str(output / "evaluation.json"),
                "exact_replay": True,
                "elapsed_seconds_including_full_replay": time.monotonic() - started,
            }
            write_new(output / "receipt.json", receipt)
            return receipt
        except BaseException as exc:
            write_new(
                output / "receipt.json",
                {
                    "status": failure_category(exc),
                    "row": row,
                    "phase": "evaluation",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "plan_sha256": verify_plan(root)["sha256"],
                },
            )
            raise
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
            if experiment is not None:
                experiment.close()
