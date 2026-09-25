"""Run five communication methods from one config, with isolated seed processes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace

from omegaconf import OmegaConf
import torch

from commstudy.experiments.bookkeeping import (
    atomic_write_json,
    atomic_write_text,
    capture_git_state,
    capture_versions,
    parameter_counts,
    read_json,
    task_runtime_contract,
    utc_now,
)
from commstudy.experiments.build import build_experiment
from commstudy.experiments.config import (
    METHODS,
    allocation,
    config_fingerprint,
    experiment_spec_from_dict,
    load_config,
    make_spec,
    resolved_experiment_dict,
)
from commstudy.experiments.metrics import ExperimentMetricsCallback, METRIC_COLUMNS
from commstudy.experiments.provenance import source_fingerprint

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_count(requested):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    allocation_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", available or 1))
    return min(requested, max(1, min(available or 1, allocation_cpus)))


@contextmanager
def experiment_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another launch is using {root}") from error
        # Workers inherit this descriptor. If the coordinator is killed, its
        # surviving workers keep the directory locked until they exit.
        yield lock.fileno()


def run_directory(root, method, seed):
    return Path(root) / method / f"seed_{seed}"


def compatible_completion(path, binding):
    status = read_json(path / "status.json", {})
    if status.get("state") != "completed":
        return False
    if status.get("binding") != binding:
        raise ValueError(f"Incompatible completed job: {path}")
    for name in ("metrics.csv", "checkpoint.pt"):
        if not (path / name).is_file() or file_sha(path / name) != status.get("sha256", {}).get(
            name
        ):
            raise ValueError(f"Completed job has a missing or changed {name}: {path}")
    if not (path / "run.out").is_file():
        raise ValueError(f"Completed job is missing run.out: {path}")
    return True


def preserve_attempt(path):
    path.mkdir(parents=True, exist_ok=True)
    previous = [p for p in path.iterdir() if p.name != "attempts"]
    if previous:
        attempts = path / "attempts"
        attempts.mkdir(exist_ok=True)
        target = attempts / f"attempt_{len(list(attempts.iterdir())) + 1:03d}"
        target.mkdir()
        for item in previous:
            item.rename(target / item.name)


def combine_results(config):
    """Only the coordinator writes experiment tables; old attempts never count."""
    root = Path(config["output_dir"])
    temporary = root / ".metrics.csv.tmp"
    runs = io.StringIO(newline="")
    summary = csv.DictWriter(runs, fieldnames=("method", "seed", "state", "frames", "error"))
    summary.writeheader()
    with temporary.open("w", encoding="utf-8", newline="") as metrics:
        writer = csv.DictWriter(metrics, fieldnames=("method", "seed", *METRIC_COLUMNS))
        writer.writeheader()
        for method, seed in allocation(config):
            path = run_directory(root, method, seed)
            status = read_json(path / "status.json", {})
            frames = 0
            if (path / "metrics.csv").exists():
                with (path / "metrics.csv").open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        if set(row) != set(METRIC_COLUMNS) or None in row.values():
                            continue  # a process may have been killed during its last write
                        frames = max(frames, int(row["frames"]))
                        writer.writerow({"method": method, "seed": seed, **row})
            summary.writerow(
                {
                    "method": method,
                    "seed": seed,
                    "state": status.get("state", "pending"),
                    "frames": frames,
                    "error": status.get("error", ""),
                }
            )
        metrics.flush()
        os.fsync(metrics.fileno())
    temporary.replace(root / "metrics.csv")
    atomic_write_text(root / "runs.csv", runs.getvalue())


def save_checkpoint(experiment, spec, path):
    checkpoint = {
        "schema_version": 1,
        "spec": resolved_experiment_dict(spec),
        "frames": experiment.total_frames,
        "policies": {g: p.state_dict() for g, p in experiment.group_policies.items()},
        "losses": {g: loss.state_dict() for g, loss in experiment.losses.items()},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_checkpoint(path, *, save_folder):
    """Rebuild actors and critics for analysis; this is not optimizer continuation."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    raw = checkpoint["spec"]
    raw["experiment"].update(
        save_folder=str(save_folder),
        restore_file=None,
        sampling_device="cpu",
        train_device="cpu",
        buffer_device="cpu",
    )
    experiment = build_experiment(experiment_spec_from_dict(raw))
    try:
        for group, state in checkpoint["losses"].items():
            experiment.losses[group].load_state_dict(state, strict=True)
        for group, state in checkpoint["policies"].items():
            experiment.group_policies[group].load_state_dict(state, strict=True)
        experiment.total_frames = checkpoint["frames"]
        return experiment
    except BaseException:
        experiment.close()
        raise


def run_job(config, method, seed, binding):
    """Execute one job in its fresh process. Status is committed last."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = run_directory(config["output_dir"], method, seed)
    path.mkdir(parents=True, exist_ok=True)
    status = {
        "method": method,
        "seed": seed,
        "binding": binding,
        "state": "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
    }
    atomic_write_json(path / "status.json", status)
    spec = make_spec(config, method, seed)
    spec = replace(spec, experiment={**spec.experiment, "save_folder": str(path / ".framework")})
    experiment = None
    try:
        callback = ExperimentMetricsCallback(path, return_groups=spec.task_config["return_groups"])

        def record_progress(record):
            if record["metric"] == "return_mean" and not record["group"]:
                status["progress"] = {
                    "frames": record["frames"],
                    "phase": record["phase"],
                    "return_mean": float(record["value"]),
                }
                atomic_write_json(path / "status.json", status)

        experiment = build_experiment(spec, callbacks=[callback])
        callback.writer.on_record = record_progress
        status["parameter_counts"] = parameter_counts(experiment)
        status["task_contract"] = task_runtime_contract(experiment)
        experiment.run()
        if experiment.total_frames != config["frames"]:
            raise RuntimeError(f"Expected {config['frames']} frames, got {experiment.total_frames}")
        if any(
            not torch.isfinite(p).all()
            for loss in experiment.losses.values()
            for p in loss.parameters()
        ):
            raise RuntimeError("Nonfinite final learned parameters")
        save_checkpoint(experiment, spec, path / "checkpoint.pt")
        status.update(
            state="completed",
            frames=experiment.total_frames,
            sha256={name: file_sha(path / name) for name in ("metrics.csv", "checkpoint.pt")},
        )
    except BaseException as error:
        status.update(
            state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        traceback.print_exc()
    finally:
        if experiment is not None:
            experiment.close()
        status["finished_at"] = utc_now()
        atomic_write_json(path / "status.json", status)
    return 0 if status["state"] == "completed" else 1


def run_experiment(config):
    root = Path(config["output_dir"])
    manifest = None
    if config["task"] == "mapdn":
        from commstudy.tasks.preparation import data_manifest, normalization_windows

        manifest = data_manifest(config)
        normalization_windows(config, manifest)
    versions = capture_versions()
    identity = {
        "config_sha256": config_fingerprint(config),
        "source_sha256": source_fingerprint(),
        "dependencies": versions,
        "runtime": {"platform": sys.platform, "machine": platform.machine(), "threads": 1},
        "data_sha256": manifest["sha256"] if manifest else None,
    }
    binding = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    with experiment_lock(root) as lock_fd:
        existing = read_json(root / "provenance.json")
        if existing is not None and existing.get("binding") != binding:
            raise ValueError(
                f"Incompatible config, code, dependencies, or data in {root}; use a new output_dir"
            )
        if existing is None:
            if any((root / name).exists() for name in (*METHODS, "config.yaml", "data")):
                raise ValueError(f"Unrecognized existing results in {root}; use a new output_dir")
            source = Path(__file__).resolve().parents[3]
            git = capture_git_state(source, root) if (source / ".git").exists() else None
            effective = {
                method: resolved_experiment_dict(make_spec(config, method, config["seeds"][0]))
                for method in METHODS
            }
            atomic_write_json(
                root / "provenance.json",
                {
                    **identity,
                    "binding": binding,
                    "created_at": utc_now(),
                    "git": git,
                    "threads_per_worker": 1,
                    "device": config["execution"]["device"],
                    "resolved_method_specs": effective,
                    "data_files": manifest["data"]["files"] if manifest else {},
                },
            )
        atomic_write_text(root / "config.yaml", OmegaConf.to_yaml(config))
        with (root / "experiment.out").open("a", buffering=1) as log:

            def report(message):
                line = f"{utc_now()} {message}"
                print(line, flush=True)
                log.write(line + "\n")

            if manifest:
                from commstudy.tasks.preparation import prepare_mapdn

                report("Preparing/checking training-only MAPDN normalization")
                prepare_mapdn(config, manifest)
            pending = []
            for method, seed in allocation(config):
                if compatible_completion(run_directory(root, method, seed), binding):
                    report(f"SKIP {method} seed={seed}: completed")
                else:
                    pending.append((method, seed))
            count = worker_count(config["execution"]["workers"])
            report(f"{len(pending)} jobs to run, {count} workers, 1 numerical thread per worker")
            active, failures, progress_seen = {}, [], {}
            env = {
                **os.environ,
                **{name: "1" for name in THREAD_ENV},
                "PYTHONUNBUFFERED": "1",
                "TQDM_DISABLE": "1",
            }
            prior_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
            stop_requested = False

            def interrupted(signum, frame):
                nonlocal stop_requested
                stop_requested = True

            for sig in prior_handlers:
                signal.signal(sig, interrupted)
            try:
                while pending or active:
                    if stop_requested:
                        raise KeyboardInterrupt("Launch interrupted")
                    while pending and len(active) < count and not stop_requested:
                        method, seed = pending.pop(0)
                        path = run_directory(root, method, seed)
                        preserve_attempt(path)
                        atomic_write_json(
                            path / "status.json", {"state": "running", "binding": binding}
                        )
                        with (path / "run.out").open("a") as out:
                            process = subprocess.Popen(
                                [
                                    sys.executable,
                                    "-m",
                                    "commstudy.experiments.runner",
                                    "--worker",
                                    str(root),
                                    method,
                                    str(seed),
                                ],
                                stdout=out,
                                stderr=subprocess.STDOUT,
                                env=env,
                                pass_fds=(lock_fd,),
                                start_new_session=True,
                            )
                        active[process] = (method, seed, path)
                        report(f"START {method} seed={seed}")
                    for process, (method, seed, path) in list(active.items()):
                        code = process.poll()
                        status = read_json(path / "status.json", {})
                        progress = status.get("progress")
                        if progress and progress_seen.get((method, seed)) != progress:
                            report(
                                f"PROGRESS {method} seed={seed} frames={progress['frames']} "
                                f"{progress['phase']}/return_mean={progress['return_mean']}"
                            )
                            progress_seen[method, seed] = progress
                        if code is None:
                            continue
                        if code != 0 or status.get("state") != "completed":
                            status.update(
                                state="failed", error=status.get("error", f"Worker exited {code}")
                            )
                            atomic_write_json(path / "status.json", status)
                            failures.append(f"{method} seed={seed}: {status['error']}")
                        report(f"{status['state'].upper()} {method} seed={seed}")
                        del active[process]
                    time.sleep(0.25)
            except BaseException:
                for process in active:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                for process, (_, _, path) in active.items():
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    status = read_json(path / "status.json", {})
                    if status.get("state") != "completed":
                        status.update(state="interrupted", error="Coordinator interrupted")
                        atomic_write_json(path / "status.json", status)
                report(
                    "Launch interrupted; completed jobs are retained. "
                    "Other attempts restart next time."
                )
                raise
            finally:
                for sig, handler in prior_handlers.items():
                    signal.signal(sig, handler)
                combine_results(config)
            for failure in failures:
                report(f"FAILED {failure}")
            report(
                f"Finished: {len(allocation(config)) - len(failures)} completed, "
                f"{len(failures)} failed"
            )
            return 1 if failures else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker":
        _, root, method, seed = argv
        config = load_config(Path(root) / "config.yaml")
        return run_job(
            config, method, int(seed), read_json(Path(root) / "provenance.json")["binding"]
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and show jobs without training"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.dry_run:
            print(
                f"{config['task']} / {config['algorithm']}: {len(allocation(config))} policies; "
                f"{config['frames']} joint environment frames each; "
                f"{worker_count(config['execution']['workers'])} workers"
            )
            print(f"Output: {config['output_dir']}")
            for method, seed in allocation(config):
                print(f"  {method:10s} seed={seed}")
            if config["task"] == "mapdn":
                from commstudy.tasks.preparation import data_manifest, normalization_windows

                normalization_windows(config, data_manifest(config))
            print("Configuration valid")
            return 0
        return run_experiment(config)
    except KeyboardInterrupt:
        return 130
    except (ValueError, TypeError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
